import streamlit as st
import whisper
import av
import os
import time
import tempfile
import pydub
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from deep_translator import GoogleTranslator
from gtts import gTTS

# --- PAGE SETUP ---
st.set_page_config(page_title="Hands-Free Interpreter", layout="wide")

# --- SESSION STATE ---
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = pydub.AudioSegment.empty()
if "last_process_time" not in st.session_state:
    st.session_state.last_process_time = time.time()

# --- LOAD WHISPER MODEL ---
@st.cache_resource
def load_whisper():
    # 'tiny' is used for speed. 
    return whisper.load_model("tiny")

model = load_whisper()

# --- HELPER: GET LANGUAGE CODES ---
try:
    LANG_CODES = GoogleTranslator().get_supported_languages(as_dict=True)
except:
    LANG_CODES = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "arabic": "ar"}

# --- AUDIO CALLBACK ---
# This collects audio from the browser continuously
def process_audio(frame: av.AudioFrame):
    sound = pydub.AudioSegment(
        data=frame.to_ndarray().tobytes(),
        sample_width=frame.format.bytes,
        frame_rate=frame.sample_rate,
        channels=len(frame.layout.channels)
    )
    st.session_state.audio_buffer += sound

# --- MAIN APP ---
def main():
    st.title("🗣️ Hands-Free Classroom Interpreter")
    st.markdown("### Auto-Detects Teacher & Translates Continuously")

    # 1. Settings
    col1, col2 = st.columns(2)
    with col1:
        st.info("Teacher Language: **Auto-Detect** (AI will guess)")
    
    with col2:
        # Target Language Selection
        display_names = [name.capitalize() for name in LANG_CODES.keys()]
        default_index = display_names.index("Spanish") if "Spanish" in display_names else 0
        target_lang_name = st.selectbox("Student hears:", display_names, index=default_index)
        target_lang_code = LANG_CODES.get(target_lang_name.lower(), "en")

    st.divider()

    # 2. Instructions
    st.warning(
        "**Setup:** Select 'CABLE Input' in Zoom. Select 'CABLE Output' in the Start button below."
    )

    # 3. Continuous Listener (WebRTC)
    ctx = webrtc_streamer(
        key="live-interpreter",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=process_audio,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # 4. Processing Loop
    status_placeholder = st.empty()
    output_placeholder = st.empty()
    audio_placeholder = st.empty()

    while ctx.state.playing:
        status_placeholder.markdown("🔴 **Listening...**")
        
        # Check if 6 seconds have passed
        now = time.time()
        if now - st.session_state.last_process_time > 6:
            
            # Get audio from buffer
            audio_chunk = st.session_state.audio_buffer
            
            # If we have at least 1 second of audio
            if len(audio_chunk) > 1000: 
                # Reset buffer and timer
                st.session_state.audio_buffer = pydub.AudioSegment.empty()
                st.session_state.last_process_time = now
                
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    audio_chunk.export(tmp_file.name, format="wav")
                    tmp_path = tmp_file.name
                
                try:
                    # A. Transcribe (Auto-Detect Language)
                    # We do NOT pass language="en" here. We let Whisper guess.
                    result = model.transcribe(tmp_path) 
                    text = result["text"].strip()
                    detected_lang = result.get("language", "unknown")
                    
                    if len(text) > 2: # Ignore empty noise
                        
                        # B. Translate
                        translator = GoogleTranslator(source='auto', target=target_lang_code)
                        trans_text = translator.translate(text)
                        
                        # C. Display
                        with output_placeholder.container():
                            st.caption(f"Detected: {detected_lang}")
                            st.markdown(f"**Teacher:** {text}")
                            st.success(f"**Translation:** {trans_text}")

                        # D. Speak
                        try:
                            tts = gTTS(text=trans_text, lang=target_lang_code)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                # This plays the audio automatically
                                audio_placeholder.audio(fp.name, format="audio/mp3", autoplay=True)
                        except Exception as e:
                            print(f"Audio Error: {e}")

                except Exception as e:
                    print(f"Error: {e}")
                
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
