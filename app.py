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
st.set_page_config(page_title="Real-Time Interpreter", layout="wide")

# --- SESSION STATE ---
# We use this to store audio while the app is running
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = pydub.AudioSegment.empty()
if "last_process_time" not in st.session_state:
    st.session_state.last_process_time = time.time()
if "transcript_history" not in st.session_state:
    st.session_state.transcript_history = []

# --- LOAD MODELS ---
# Use @st.cache_resource so we only load the model once (saves speed)
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny") # 'tiny' is essential for real-time speed

model = load_whisper()

# --- AUDIO CALLBACK (The "Ear") ---
# This function runs in the background and grabs audio chunks from the browser
def process_audio(frame: av.AudioFrame):
    sound = pydub.AudioSegment(
        data=frame.to_ndarray().tobytes(),
        sample_width=frame.format.bytes,
        frame_rate=frame.sample_rate,
        channels=len(frame.layout.channels)
    )
    # Add new sound to our buffer
    st.session_state.audio_buffer += sound

# --- MAIN APP ---
def main():
    st.title("⚡ Real-Time Classroom Interpreter")
    st.markdown("This app listens continuously. **Click Start** and it will translate every few seconds.")

    # 1. Settings
    col1, col2 = st.columns(2)
    with col1:
        st.info("Teacher is speaking: **English**")
    with col2:
        # Target Language
        try:
            langs = GoogleTranslator().get_supported_languages()
            target_lang = st.selectbox("Translate to:", [l.capitalize() for l in langs], index=langs.index("spanish"))
        except:
            target_lang = "Spanish"

    st.divider()

    # 2. The "Live" Connection
    # This creates the continuous WebRTC connection
    ctx = webrtc_streamer(
        key="live-interpreter",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=process_audio,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # 3. The Processing Loop
    # If the streamer is active (Green light), run this loop
    status_placeholder = st.empty()
    output_placeholder = st.empty()

    while ctx.state.playing:
        status_placeholder.markdown("🔴 **Listening...** (Do not close this tab)")
        
        # Check if 5 seconds have passed
        now = time.time()
        if now - st.session_state.last_process_time > 5:
            
            # Grab the audio buffer
            audio_chunk = st.session_state.audio_buffer
            
            # Only process if we have audio
            if len(audio_chunk) > 0:
                # Reset buffer and timer immediately (so we don't miss new words)
                st.session_state.audio_buffer = pydub.AudioSegment.empty()
                st.session_state.last_process_time = now
                
                # Save to temp file for Whisper
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    audio_chunk.export(tmp_file.name, format="wav")
                    tmp_path = tmp_file.name
                
                try:
                    # A. Transcribe
                    result = model.transcribe(tmp_path)
                    text = result["text"].strip()
                    
                    if len(text) > 1: # Ignore silence/noise
                        # B. Translate
                        translator = GoogleTranslator(source='auto', target=target_lang.lower())
                        trans_text = translator.translate(text)
                        
                        # C. Add to history
                        st.session_state.transcript_history.append((text, trans_text))
                        
                        # D. Speak (Autoplay)
                        tts = gTTS(text=trans_text, lang=target_lang.lower())
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                            tts.save(fp.name)
                            st.audio(fp.name, format="audio/mp3", start_time=0)

                except Exception as e:
                    print(f"Error: {e}")
                
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # Display History (Reverse order so newest is on top)
        with output_placeholder.container():
            for original, translated in reversed(st.session_state.transcript_history[-5:]):
                st.markdown(f"**Teacher:** {original}")
                st.success(f"**{target_lang}:** {translated}")
                st.divider()
        
        # Sleep briefly to save CPU
        time.sleep(0.5)

if __name__ == "__main__":
    main()
