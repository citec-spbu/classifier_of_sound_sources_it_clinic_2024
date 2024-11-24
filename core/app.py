import streamlit as st
from io import BytesIO
import os
import librosa.display
import matplotlib.pyplot as plt
from preproccessing import AudioPreprocessor 

st.title("Анализ аудио и классификация")

# загрузка файла
uploaded_file = st.file_uploader("Загрузите аудиофайл", type=["wav"])

if uploaded_file is not None:
    
    audio_data, sr = librosa.load(BytesIO(uploaded_file.read()))
    ap = AudioPreprocessor(audio=audio_data, sr=sr)

    # визуализация
    with st.expander("Оригинальное аудио"):
        ap.display_waveform()

    detected_env = ap._analyze_environment()
    st.write(f"Наиболее вероятная среда - {detected_env}")

    # преобразования
    transform_mode = st.selectbox("Режим преобразования",
                                  ["Auto", "Custom"])
    if transform_mode == "Auto":
        ap = ap.preprocessing_auto_pipeline(environment=detected_env)
    else:
        methods = st.multiselect("Методы предобработки",
                                 ["trim", "preemphasis", "normalize", 
                                  "equalize", "remove_noise"])
        ap = ap.preprocessing_custom_pipeline(methods)
    
    transformed_audio_filepath = "data/transformed_audio.wav"
    ap.save_audio(transformed_audio_filepath)
    st.success(f"Сохранено как {transformed_audio_filepath}")
    st.audio(transformed_audio_filepath)
    
    # классификация и предсказание
    # model = load_model()
    # features = ap.extract_features()
    # predictions = model.predict(features)
    # st.write("Топ 5 наиболее вероятных меток аудио:")
    # for idx, prediction in enumerate(predictions):
    #     st.write(f"#{idx+1}: {prediction}")
    
    os.remove(transformed_audio_filepath)