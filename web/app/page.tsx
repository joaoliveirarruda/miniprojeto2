import Demo from "@/components/Demo";
import results from "../../results/metrics.json";

const REPO = "https://github.com/joaoliveirarruda/miniprojeto2";
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const metrics: Record<string, { accuracy: number; macro_f1: number }> = results;

const ROWS = [
  { key: "cnn_leaky", name: "Original CNN, original evaluation", note: "augmented copies leak into test" },
  { key: "logreg", name: "Logistic regression (hand-crafted features)", note: "baseline" },
  { key: "cnn", name: "Original CNN, honest evaluation", note: "same model, unseen speakers" },
  { key: "wav2vec2", name: "Fine-tuned wav2vec2 (this demo)", note: "raw audio in, unseen speakers" },
] as const;

export default function Home() {
  return (
    <main>
      <header className="hero">
        <p className="eyebrow">Speech emotion recognition</p>
        <h1>How do you sound?</h1>
        <p className="lede">
          Talk into your microphone and a neural network guesses the emotion in your voice, live. It runs entirely in your browser: your
          audio never leaves your device.
        </p>
      </header>

      <Demo />

      <section>
        <h2>How it works</h2>
        <ol className="steps">
          <li>
            <strong>Capture.</strong> Your mic is recorded into a rolling 3-second buffer. Every half second, if you're speaking, that window
            is resampled to 16 kHz.
          </li>
          <li>
            <strong>Classify.</strong> A Web Worker runs a fine-tuned <a href="https://huggingface.co/facebook/wav2vec2-base">wav2vec2</a>{" "}
            model (int8 ONNX) on the raw waveform with transformers.js, and returns a probability for each of 8 emotions.
          </li>
          <li>
            <strong>Smooth.</strong> Predictions are averaged over time, so the result follows your tone instead of flickering.
          </li>
        </ol>
      </section>

      <section>
        <h2>Results</h2>
        <p>
          Scored on 6 actors the models never heard during training (360 clips). The first row is how the original version of this
          project was evaluated, and shows why that number couldn't be trusted.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Accuracy</th>
                <th scope="col">Macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.filter(({ key }) => key in metrics).map(({ key, name, note }) => (
                <tr key={key} className={key === "wav2vec2" ? "highlight" : undefined}>
                  <td>
                    {name}
                    <br />
                    <span className="muted">{note}</span>
                  </td>
                  <td>{pct(metrics[key].accuracy)}</td>
                  <td>{pct(metrics[key].macro_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">Chance level is 12.5%. Full analysis, notebooks and training code are in the repository.</p>
      </section>

      <section>
        <h2>Limitations</h2>
        <ul>
          <li>
            Trained on <a href="https://zenodo.org/records/1188976">RAVDESS</a>: 24 North American actors reading two English sentences in a
            studio. Spontaneous speech, other languages and noisy mics are much harder for it.
          </li>
          <li>Acted emotions are exaggerated, so subtle real-life emotion is often read as neutral or calm.</li>
          <li>It hears tone, not words: “I'm so happy” said flatly will not sound happy to it.</li>
        </ul>
      </section>

      <footer>
        <a href={REPO}>Source on GitHub</a> · RAVDESS by Livingstone &amp; Russo (2018), CC BY-NC-SA 4.0 · Non-commercial demo
      </footer>
    </main>
  );
}
