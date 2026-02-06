import streamlit as st
import whisper
import os
import tempfile
from deep_translator import GoogleTranslator
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# --- PAGE SETUP ---
st.set_page_config(page_title="Classroom Interpreter", layout="wide")

# --- HELPER: GET LANGUAGE CODES ---
# This fixes the error. We get a dictionary like {'arabic': 'ar', 'spanish': 'es'}
try:
    LANG_CODES = GoogleTranslator().get_supported_languages(as_dict=True)
except:
    # Backup in case internet fails momentarily
    LANG_CODES = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "arabic": "ar"}

def main():
    st.title("👨‍🏫 Classroom Live Interpreter")
    st.markdown("### Step 1: Select Your Language")

    # --- 1. Settings ---
    col1, col2 = st.columns(2)
    with col1:
        # Teacher's side (Source)
        st.info("Teacher is speaking: **English**")
        
    with col2:
        # Student's side (Target)
        # We create a list of nice names (Capitalized) for the dropdown
        display_names = [name.capitalize() for name in LANG_CODES.keys()]
        
        # Default to Spanish if available
        default_index = display_names.index("Spanish") if "Spanish" in display_names else 0
        
        target_lang_name = st.selectbox(
            "I want to hear:",
            display_names,
            index=default_index
        )
        
        # KEY FIX: Get the 2-letter code (e.g., 'ar') for the selected name
        target_lang_code = LANG_CODES.get(target_lang_name.lower(), "en")

    st.divider()

    # --- 2. The Listener ---
    st.markdown("### Step 2: Listen to Teacher")
    st.info(
        "**Instructions:**\n"
        "1. In Zoom, set Speaker to **'CABLE Input'** (or BlackHole).\n"
        "2. Click Start below, then select **'CABLE Output'** as the microphone."
    )

    # The Recording Component
    audio_data = mic_recorder(
        start_prompt="🔴 Start Listening",
        stop_prompt="⏹️ Stop & Translate",
        key="recorder",
        format="wav",
        use_container_width=True
    )

    # --- 3. Processing & Output ---
    if audio_data is not None:
        st.divider()
        audio_bytes = audio_data['bytes']
        
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # A. Transcribe (English)
            with st.spinner("🎧 Transcribing teacher's voice..."):
                # Use 'tiny' for speed
                model = whisper.load_model("tiny")
                result = model.transcribe(tmp_path)
                original_text = result["text"]

            if original_text.strip():
                # B. Translate
                with st.spinner("🌍 Translating..."):
                    # Use the "key fix" code from above
                    translator = GoogleTranslator(source='auto', target=target_lang_code)
                    translated_text = translator.translate(original_text)

                # C. Display & Speak
                col_orig, col_trans = st.columns(2)
                
                with col_orig:
                    st.markdown("**Teacher Said (English):**")
                    st.caption(original_text)
                
                with col_trans:
                    st.markdown(f"**Translation ({target_lang_name}):**")
                    st.success(translated_text)
                    
                    # Generate Audio
                    try:
                        # KEY FIX: Use the 2-letter code (e.g., 'ar') here
                        tts = gTTS(text=translated_text, lang=target_lang_code)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_fp:
                            tts.save(audio_fp.name)
                            # Auto-play the translation
                            st.audio(audio_fp.name, format="audio/mp3", start_time=0)
                    except Exception as e:
                        st.error(f"Audio error: {e}")

            else:
                st.warning("⚠️ No sound detected. Did you select 'CABLE Output'?")

        except Exception as e:
            st.error(f"Error: {e}")
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    main()
