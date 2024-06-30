import streamlit as st
from streamlit_chat import message
from streamlit_mic_recorder import mic_recorder
import openai
from openai import OpenAI
import whisper
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

model = whisper.load_model("base")

# Function to convert audio to text w/ Whisper
def convert_audio_to_text(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_file_path = tmp_file.name

    audio = whisper.load_audio(tmp_file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en")
    result = whisper.decode(model, mel, options)
    return result.text

# Function to get response from GPT
def get_response_from_model(user_input):
    print(user_input)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    print(response)
    return response.choices[0].message.content

# Function to get model response
def get_model_response(user_input):
    return get_response_from_model(user_input)

# Main app logic
st.title("Language Chat")

def recording_callback():
    if st.session_state.my_recorder_output:
        audio_bytes = st.session_state.my_recorder_output['bytes']
        audio_text = convert_audio_to_text(audio_bytes)
        message(audio_text, is_user=True)
        model_response = get_model_response(audio_text)
        message(model_response, is_user=False)

mic_recorder(key='my_recorder', callback=recording_callback)



user_input = st.text_input("You: ")
if user_input:
    model_response = get_model_response(user_input)
    message(user_input, is_user=True)
    message(model_response, is_user=False)
