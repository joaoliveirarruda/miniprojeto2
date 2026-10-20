"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { decodeTo16k, mean, resampleTo16k, RingBuffer, rms, SILENCE_RMS, smooth, top, WINDOW_SECONDS, windows } from "@/lib/audio";
import { SAMPLE_RATE, type OutMessage, type Score } from "@/lib/inference";

export const EMOTION_META: Record<string, { emoji: string; color: string }> = {
  angry: { emoji: "😠", color: "var(--angry)" },
  calm: { emoji: "😌", color: "var(--calm)" },
  disgust: { emoji: "🤢", color: "var(--disgust)" },
  fear: { emoji: "😨", color: "var(--fear)" },
  happy: { emoji: "😄", color: "var(--happy)" },
  neutral: { emoji: "😐", color: "var(--neutral)" },
  sad: { emoji: "😢", color: "var(--sad)" },
  surprise: { emoji: "😲", color: "var(--surprise)" },
};

const SAMPLES = Object.keys(EMOTION_META);
const TICK_MS = 500;

type ModelStatus = { state: "loading"; progress: number } | { state: "ready" } | { state: "error"; message: string };
type Live = "idle" | "starting" | "listening" | "denied" | "error";
type FileResult = { name: string; scores: Score[]; timeline: { start: number; label: string }[] };

function useClassifier() {
  const worker = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, (m: OutMessage) => void>());
  const nextId = useRef(0);
  const [status, setStatus] = useState<ModelStatus>({ state: "loading", progress: 0 });

  useEffect(() => {
    const w = new Worker(new URL("../lib/worker.ts", import.meta.url), { type: "module" });
    w.onmessage = ({ data: m }: MessageEvent<OutMessage>) => {
      if (m.type === "progress" && m.status === "progress" && m.file?.endsWith(".onnx")) {
        setStatus({ state: "loading", progress: m.progress ?? 0 });
      } else if (m.type === "ready") {
        setStatus({ state: "ready" });
      } else if (m.type === "error" && m.id === undefined) {
        setStatus({ state: "error", message: m.message });
      } else if ("id" in m && m.id !== undefined) {
        pending.current.get(m.id)?.(m);
        pending.current.delete(m.id);
      }
    };
    w.postMessage({ type: "load" });
    worker.current = w;
    return () => w.terminate();
  }, []);

  const classify = useCallback(
    (audio: Float32Array) =>
      new Promise<{ scores: Score[]; ms: number }>((resolve, reject) => {
        const id = ++nextId.current;
        pending.current.set(id, (m) => (m.type === "result" ? resolve(m) : reject(new Error(m.type === "error" ? m.message : m.type))));
        worker.current!.postMessage({ type: "classify", id, audio, sampleRate: SAMPLE_RATE });
      }),
    [],
  );

  return { status, classify };
}

export default function Demo() {
  const { status, classify } = useClassifier();
  const ready = status.state === "ready";
  const [supported, setSupported] = useState(true);
  const [live, setLive] = useState<Live>("idle");
  const [speaking, setSpeaking] = useState(false);
  const [scores, setScores] = useState<Score[] | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [file, setFile] = useState<FileResult | null>(null);
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const stopRef = useRef<() => void>(() => {});
  const levelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSupported(typeof AudioWorkletNode !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof WebAssembly !== "undefined");
    return () => stopRef.current();
  }, []);

  async function start() {
    setLive("starting");
    setScores(null);
    // Created before any await, while the click still counts as a user gesture: otherwise
    // a slow permission prompt leaves the context suspended (autoplay policy) and silent.
    const ctx = new AudioContext();
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } });
    } catch {
      ctx.close();
      setLive("denied");
      return;
    }
    try {
      await ctx.resume();
      await ctx.audioWorklet.addModule("/recorder-worklet.js");
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "recorder");
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(node).connect(mute).connect(ctx.destination);

      const ring = new RingBuffer(WINDOW_SECONDS * ctx.sampleRate);
      const recent = new RingBuffer(ctx.sampleRate); // last second, for the silence gate
      node.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
        ring.push(data);
        recent.push(data);
        if (levelRef.current) levelRef.current.style.transform = `scaleX(${Math.min(1, rms(data) * 8)})`;
      };

      let busy = false;
      const timer = setInterval(async () => {
        if (busy) return; // drop this tick instead of queueing behind a slow inference
        if (rms(recent.read()) < SILENCE_RMS) {
          setSpeaking(false);
          return;
        }
        busy = true;
        try {
          const result = await classify(await resampleTo16k(ring.read(), ctx.sampleRate));
          setSpeaking(true);
          setScores((prev) => smooth(prev, result.scores));
          setLatency(result.ms);
        } finally {
          busy = false;
        }
      }, TICK_MS);

      stopRef.current = () => {
        clearInterval(timer);
        stream.getTracks().forEach((t) => t.stop());
        ctx.close();
        if (levelRef.current) levelRef.current.style.transform = "scaleX(0)";
      };
      setLive("listening");
    } catch {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      setLive("error");
    }
  }

  function stop() {
    stopRef.current();
    stopRef.current = () => {};
    setLive("idle");
    setSpeaking(false);
  }

  async function analyze(name: string, data: ArrayBuffer) {
    setAnalyzing(name);
    try {
      const audio = await decodeTo16k(data);
      const all = windows(audio);
      const voiced = all.filter((w) => rms(w.audio) >= SILENCE_RMS);
      const results = [];
      for (const w of voiced.length ? voiced : all) results.push({ start: w.start, scores: (await classify(w.audio)).scores });
      setFile({ name, scores: mean(results.map((r) => r.scores)), timeline: results.map((r) => ({ start: r.start, label: top(r.scores).label })) });
    } catch {
      setFile(null);
      setFileError(`Couldn't read "${name}". Try a WAV, MP3 or OGG file.`);
    } finally {
      setAnalyzing(null);
    }
  }

  async function playSample(emotion: string) {
    const url = `/samples/${emotion}.wav`;
    new Audio(url).play().catch(() => {});
    setFileError(null);
    await analyze(`${emotion}.wav (RAVDESS, unseen speaker)`, await (await fetch(url)).arrayBuffer());
  }

  const listening = live === "listening";
  const best = scores && speaking ? top(scores) : null;

  return (
    <section className="demo" aria-labelledby="demo-title">
      <h2 id="demo-title" className="visually-hidden">Live demo</h2>

      <ModelBadge status={status} latency={latency} />

      {!supported ? (
        <p className="notice">This browser can't capture audio for analysis. Try a recent Chrome, Edge, Firefox or Safari.</p>
      ) : (
        <div className="live">
          <button className="primary" onClick={listening ? stop : start} disabled={!ready || live === "starting"} aria-pressed={listening}>
            {listening ? "Stop listening" : live === "starting" ? "Starting…" : "Start listening"}
          </button>
          <div className="level" aria-hidden="true">
            <div ref={levelRef} className="level-fill" />
          </div>
          <p className="live-status" aria-live="polite">
            {live === "denied" && "Microphone access was blocked. Allow it in your browser's site settings and try again."}
            {live === "error" && "Couldn't start the microphone."}
            {listening && !speaking && "Listening… say something with feeling."}
            {listening && best && (
              <>
                Sounds <strong>{best.label}</strong>
              </>
            )}
            {live === "idle" && ready && "Speak in English, like an actor: the model learned from acted speech."}
          </p>
          <div className="headline" aria-hidden="true">
            <span className="emoji">{best ? EMOTION_META[best.label].emoji : "🎙️"}</span>
            <span className="label">{best ? best.label : listening ? "waiting for speech" : "—"}</span>
          </div>
          <Bars scores={scores} dim={!speaking} />
        </div>
      )}

      <div className="files">
        <h3>Or try a recording</h3>
        <div className="samples" role="group" aria-label="RAVDESS sample clips">
          {SAMPLES.map((e) => (
            <button key={e} className="chip" onClick={() => playSample(e)} disabled={!ready || !!analyzing}>
              {EMOTION_META[e].emoji} {e}
            </button>
          ))}
        </div>
        <label className="upload">
          <input
            type="file"
            accept="audio/*"
            disabled={!ready || !!analyzing}
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) {
                setFileError(null);
                await analyze(f.name, await f.arrayBuffer());
              }
            }}
          />
          <span>Upload an audio file</span>
        </label>
        <div aria-live="polite">
          {analyzing && <p className="muted">Analyzing {analyzing}…</p>}
          {fileError && <p className="notice">{fileError}</p>}
          {file && !analyzing && (
            <div className="file-result">
              <p>
                <strong>{file.name}</strong>: mostly <strong>{top(file.scores).label}</strong> {EMOTION_META[top(file.scores).label].emoji}
              </p>
              <Bars scores={file.scores} />
              {file.timeline.length > 1 && (
                <ol className="timeline" aria-label="Emotion per 3-second window">
                  {file.timeline.map((t) => (
                    <li key={t.start} style={{ borderColor: EMOTION_META[t.label].color }}>
                      <span className="muted">{t.start.toFixed(1)}s</span> {EMOTION_META[t.label].emoji} {t.label}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function Bars({ scores, dim = false }: { scores: Score[] | null; dim?: boolean }) {
  const list = scores ?? SAMPLES.map((label) => ({ label, score: 0 }));
  return (
    <ul className={`bars${dim ? " dim" : ""}`}>
      {list.map((s) => (
        <li key={s.label}>
          <span className="bar-label">{s.label}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ transform: `scaleX(${s.score})`, background: EMOTION_META[s.label].color }} />
          </span>
          <span className="bar-value">{Math.round(s.score * 100)}%</span>
        </li>
      ))}
    </ul>
  );
}

function ModelBadge({ status, latency }: { status: ModelStatus; latency: number | null }) {
  if (status.state === "error") return <p className="notice">Couldn't load the model: {status.message}</p>;
  if (status.state === "loading")
    return (
      <div className="model-loading">
        <p className="muted">Downloading the model to your browser (once, then it's cached)… {Math.round(status.progress)}%</p>
        <progress max={100} value={status.progress} aria-label="Model download progress" />
      </div>
    );
  return <p className="muted model-ready">Model loaded · runs on your device{latency !== null && ` · ${Math.round(latency)} ms per window`}</p>;
}
