import streamlit as st
from streamlit_chat import message
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import openai
import whisper
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_key

model = whisper.load_model("text-embedding-3-large")

# Function to convert audio to text w/ Whisper
def convert_audio_to_text(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data.tobytes())
        tmp_file_path = tmp_file.name

    result = model.transcribe(tmp_file_path)
    return result['text']

# Function to get response from GPT
def get_response_from_model(user_input):
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=user_input,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to get model response
def get_model_response(user_input):
    return get_response_from_model(user_input)

# Main app logic
st.title("Language Chat")

# Real-time audio  
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={
            "audio": True,
            "video": False,
        },
    ),
)

if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    if audio_frames:
        audio_data = audio_frames[0].to_ndarray()
        audio_text = convert_audio_to_text(audio_data)
        message(audio_text, is_user=True)

user_input = st.text_input("You: ")
if user_input:
    model_response = get_model_response(user_input)
    message(user_input, is_user=True)
    message(model_response, is_user=False)