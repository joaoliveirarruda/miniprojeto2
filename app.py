import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "/home/joaopc/Documents/gago/trilha/miniprojeto2/models/audio_emotion_model.keras"  # Example
SCALER_PATH = "/home/joaopc/Documents/gago/trilha/miniprojeto2/models/scaler.pkl"                # Example

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features.extend(zcr)

    # Chroma STFT
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    features.extend(chroma)

    # MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    features.extend(mfccs)

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features.extend(rms)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.title('🎵Detector de emoções em áudio')
st.write('Envie um arquivo de áudio para análise!')
st.divider()

with st.sidebar:
    st.header("Sobre")
    st.write('Aplicativo feito para projeto do Trilha')


# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    temp_audio_path = "temp_audio_file"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Reproduzir o áudio enviado
    st.audio(uploaded_file, format='audio/wav') 

    # Extrair features
    features = extract_features(temp_audio_path)

    # Normalizar os dados com o scaler treinado
    features = scaler.transform(features)

    # Ajustar formato para o modelo
    features = features.reshape(1, 162, 1)

    # Fazer a predição
    predictions = model.predict(features)
    predicted_emotion = EMOTIONS[np.argmax(predictions)]

    # Exibir o resultado
    tab1, tab2 = st.tabs(["Gráfico de Barras", "Barra de progresso"])
    
    with tab1:
        st.subheader('🎭Resultado da Análise')
        st.write(f' **{predicted_emotion.upper()}**')
        st.subheader('Probabilidades das Emoções')
        probalidade = {emotion: float(pred) for emotion, pred in zip(EMOTIONS, predictions[0])}
        st.bar_chart(probalidade)
    
    with tab2:
        st.subheader('Detalhes da Análise')
        st.write('Probabilidades para cada emoção:')
        for emotion, prob in zip(EMOTIONS, predictions[0]):
            label_text = (f'{emotion}: {prob:.2f}')
            st.progress(int(prob * 100), text=label_text)


    
     # Remover o arquivo temporário
    os.remove(temp_audio_path)
    