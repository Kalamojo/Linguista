import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from tts import recording_callback

# import language_tool_python
# tool = language_tool_python.LanguageTool('en')

st.set_page_config(page_title="Linguista - Language learning thing")
st.title("Linguista")
#print(tool.check("This are bad."))

mic_recorder(key='my_recorder', callback=recording_callback)
