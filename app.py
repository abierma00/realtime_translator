import streamlit as st
import whisper
import os
import tempfile
from whisper.tokenizer import LANGUAGES
from deep_translator import GoogleTranslator
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder # <--- NEW: Microphone access

# --- FFmpeg Check ---
try:
    import imageio_ffmpeg
    ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

# Helper: Map language codes
WHISPER_LANGUAGES = {v.capitalize(): k for k, v in LANGUAGES.items()}

def main():
    st.title("🗣️ Live Classroom Interpreter")
    st.markdown("Use this app to translate your teacher's voice in real-time.")

    # --- 1. Settings ---
    with st.expander("⚙️ Language Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # Source is usually English for the teacher
            source_options = ["Auto-Detect"] + list(WHISPER_LANGUAGES.keys())
            source_lang_name = st.selectbox("Teacher's Language", source_options, index=source_options.index("English"))
        
        with col2:
            # Target is the student's home language
            translator_langs = GoogleTranslator().get_supported_languages()
            translator_langs = [lang.capitalize() for lang in translator_langs]
            target_lang_name = st.selectbox(
                "My Home Language",
                translator_langs,
                index=translator_langs.index("Spanish") if "Spanish" in translator_langs else 0
            )

    # --- 2. The "Listen" Button ---
    st.divider()
    st.subheader("🎤 Press to Listen")
    st.info("Click the button below to start recording. Click it again to stop.")

    # This is the special microphone component
    # It returns a dictionary with 'bytes' (the audio data) when recording stops
    audio_data = mic_recorder(
        start_prompt="🔴 Start Listening",
        stop_prompt="⏹️ Stop & Translate",
        key="recorder",
        format="wav",
        use_container_width=True
    )

    # --- 3. Processing ---
    if audio_data is not None:
        # We have audio!
        audio_bytes = audio_data['bytes']
        
        # Save audio bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # --- PHASE 1: TRANSCRIBE ---
            # We use 'tiny' model here for SPEED. 'base' is more accurate but slower.
            with st.spinner("🎧 Listening & Transcribing..."):
                model = whisper.load_model("tiny")
                
                whisper_lang_code = None
                if source_lang_name != "Auto-Detect":
                    whisper_lang_code = WHISPER_LANGUAGES[source_lang_name]

                result = model.transcribe(tmp_path, language=whisper_lang_code)
                original_text = result["text"]

            # --- PHASE 2: TRANSLATE ---
            if original_text.strip():
                with st.spinner("🌍 Translating..."):
                    target_code = target_lang_name.lower()
                    translator = GoogleTranslator(source='auto', target=target_code)
                    translated_text = translator.translate(original_text)

                # --- PHASE 3: DISPLAY & SPEAK ---
                
                # Layout: Original vs Translated
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Teacher Said:**")
                    st.info(original_text)
                with c2:
                    st.markdown(f"**Translation ({target_lang_name}):**")
                    st.success(translated_text)
                
                # Auto-play audio
                try:
                    tts = gTTS(text=translated_text, lang=target_code)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_fp:
                        tts.save(audio_fp.name)
                        st.audio(audio_fp.name, format="audio/mp3", start_time=0)
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")
            else:
                st.warning("No speech detected. Please try again.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    main()
