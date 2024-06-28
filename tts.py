import streamlit as st
from openai import OpenAI
import io
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

def recording_callback():
    if st.session_state.my_recorder_output:
        audio_bytes = st.session_state.my_recorder_output['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        audio_bio.name = 'audio.webm'
        print('doin whisper stuff')
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bio,
            language="en"
        )
        st.write(transcription.text)
