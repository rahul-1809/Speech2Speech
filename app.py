import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import gtts
import numpy as np
import time
from io import BytesIO
import jiwer

# Configure Gemini API
genai.configure(api_key="AIzaSyB__i49RY6uQJI2yqIkQhVoldC0o6_V3QE")  
model = genai.GenerativeModel('gemini-1.5-flash')

# Supported Languages
LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi', 'ta': 'Tamil',
    'ur': 'Urdu', 'gu': 'Gujarati', 'ml': 'Malayalam', 'kn': 'Kannada', 'pa': 'Punjabi',
    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'zh-CN': 'Chinese (Simplified)',
    'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'pt': 'Portuguese', 'ar': 'Arabic',
    'it': 'Italian', 'nl': 'Dutch', 'tr': 'Turkish'
}

recognizer = sr.Recognizer()

def translate_text(text, source_lang, target_lang):
    start_time = time.perf_counter()
    
    try:
        prompt = (
            f"Translate the following {LANGUAGES[source_lang]} sentence to {LANGUAGES[target_lang]}:\n\n"
            f"Only return the translated text.\n\n"
            f"Sentence: {text}"
        )
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
    except Exception as e:
        translated_text = f"Translation error: {str(e)}"
    
    end_time = time.perf_counter()
    return translated_text, end_time - start_time

def text_to_speech(text, lang):
    start_time = time.perf_counter()
    
    try:
        tts = gtts.gTTS(text, lang=lang)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
    except Exception as e:
        return None, None
    
    end_time = time.perf_counter()
    return audio_buffer, end_time - start_time

def record_audio():
    with sr.Microphone() as source:
        st.info("Recording... Speak now!")
        start_time = time.perf_counter()
        audio_data = recognizer.listen(source)
        end_time = time.perf_counter()
    return audio_data, end_time - start_time

def speech_to_text(audio_data, lang):
    start_time = time.perf_counter()
    
    try:
        text = recognizer.recognize_google(audio_data, language=lang)
    except sr.UnknownValueError:
        text = "Could not understand audio"
    except sr.RequestError as e:
        text = f"Speech recognition error: {str(e)}"
    
    end_time = time.perf_counter()
    return text, end_time - start_time

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis) * 100

def main():
    st.title("🎙️ Real-Time Speech Translator")
    
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Input Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    with col2:
        target_lang = st.selectbox("Target Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

    reference_text = st.text_area("Enter correct text (for WER testing)", "")
    
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.metrics = {}

    if st.button("🎤 Start Recording"):
        st.session_state.audio_data, audio_latency = record_audio()
        
        if st.session_state.audio_data:
            text, stt_latency = speech_to_text(st.session_state.audio_data, source_lang)
            translated, translation_latency = translate_text(text, source_lang, target_lang)
            audio_file, tts_latency = text_to_speech(translated, target_lang)
            
            st.session_state.metrics['Audio Latency'] = audio_latency
            st.session_state.metrics['STT Latency'] = stt_latency
            st.session_state.metrics['Translation Latency'] = translation_latency
            st.session_state.metrics['TTS Latency'] = tts_latency
            st.session_state.metrics['Total Latency'] = sum(filter(None, [audio_latency, stt_latency, translation_latency, tts_latency]))
            
            if reference_text.strip():
                st.session_state.metrics['WER'] = calculate_wer(reference_text, text)
            
            st.subheader("Original Text")
            st.write(f"({LANGUAGES[source_lang]}) {text}")
            
            st.subheader("Translated Text")
            st.write(f"({LANGUAGES[target_lang]}) {translated}")
            
            if audio_file:
                st.audio(audio_file, format='audio/mp3')
            
            st.subheader("Performance Metrics")
            for key, value in st.session_state.metrics.items():
                st.write(f"**{key}:** {value:.3f} sec" if isinstance(value, float) else f"**{key}:** {value}")

if __name__ == "__main__":
    main()
