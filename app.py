import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import gtts
import numpy as np
import pyaudio
import wave
from io import BytesIO
import uuid

# Configure Gemini API
genai.configure(api_key="AIzaSyB__i49RY6uQJI2yqIkQhVoldC0o6_V3QE")  # Replace with your API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Supported languages
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'bn': 'Bengali'
}

# Speech recognizer setup
recognizer = sr.Recognizer()

def translate_text(text, source_lang, target_lang):
    """Translate text using Gemini with strict control."""
    try:
        prompt = (
            f"Translate the following {LANGUAGES[source_lang]} sentence to {LANGUAGES[target_lang]}.\n\n"
            f"Only return the translated text without any explanations or additional words.\n\n"
            f"Sentence: {text}"
        )
        response = model.generate_content(prompt)
        translated_text = response.text.strip()

        # Ensure no extra words are included
        if ":" in translated_text:  
            translated_text = translated_text.split(":", 1)[-1].strip()

        return translated_text
    except Exception as e:
        return f"Translation error: {str(e)}"

def text_to_speech(text, lang):
    """Generate audio from text"""
    try:
        tts = gtts.gTTS(text, lang=lang)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def record_audio():
    """Records audio using PyAudio and saves as a WAV file"""
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16-bit format
    channels = 1
    sample_rate = 44100
    duration = 5  # Duration in seconds
    file_path = "recorded_audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    st.info("Recording... Speak now!")

    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as WAV
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    return file_path

def main():
    st.title("🎙️ Live Speech Translator")

    # Language selection
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Input Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    with col2:
        target_lang = st.selectbox("Target Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

    # Initialize session state variables
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_data = None
        st.session_state.translation = ""
        st.session_state.original_text = ""
        st.session_state.audio_file = None  # Store TTS audio file

    # Recording controls
    if not st.session_state.recording:
        if st.button("🎤 Start Recording"):
            st.session_state.recording = True
            st.session_state.audio_data = None
            st.session_state.translation = ""
            st.session_state.original_text = ""
            st.session_state.audio_file = None

    if st.session_state.recording:
        audio_path = record_audio()
        st.session_state.audio_data = audio_path
        st.session_state.recording = False  # Automatically stop recording

    if st.session_state.audio_data:
        if st.button("⏹️ Stop Recording"):
            try:
                with sr.AudioFile(st.session_state.audio_data) as source:
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language=source_lang)
                st.session_state.original_text = text

                # Translate text
                translated = translate_text(text, source_lang, target_lang)
                st.session_state.translation = translated

                # Convert translation to speech
                st.session_state.audio_file = text_to_speech(translated, target_lang)

            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display results
    if st.session_state.original_text:
        st.subheader("Original Text")
        st.write(f"({LANGUAGES[source_lang]}) {st.session_state.original_text}")

    if st.session_state.translation:
        st.subheader("Translated Text")
        st.write(f"({LANGUAGES[target_lang]}) {st.session_state.translation}")

        # Play translation audio
        if st.session_state.audio_file:
            st.audio(st.session_state.audio_file, format='audio/mp3')

            # Download button
            st.download_button(
                label="Download Translation Audio",
                data=st.session_state.audio_file,
                file_name=f"translation_{uuid.uuid4().hex}.mp3",
                mime="audio/mp3"
            )

if __name__ == "__main__":
    main()
