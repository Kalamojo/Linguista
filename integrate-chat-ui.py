import streamlit as st
from streamlit_chat import message
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import whisper
# import language_tool_python
import tempfile
import json
import os
from dotenv import load_dotenv
import io
# import ffmpeg
# import numpy as np

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


model = whisper.load_model("base")
# tool = language_tool_python.LanguageToolPublicAPI(language_code)

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

# Function to get model response
def get_model_response(user_input, language):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": (
                f"1. Respond to the user's dialogue in {language}. Ensure your response encourages the user to continue the conversation by asking simple and relevant follow-up questions, allowing them to lead the discussion.\n"
                "2. After providing the response in {language}, generate a response in English that includes:\n"
                "  a. A summary of any grammatical errors identified by LanguageTool in the user's input.\n"
                "  b. A simple explanation of why each error is incorrect in the context of {language} grammar.\n"
                "  c. Detailed examples showing the correct usage in the context of the conversation.\n"
                "Make sure to clearly separate the {language} response from the English explanation. Focus on natural speech patterns in {language}, unless the learner explicitly requests a different focus. Ensure feedback is concise and split into manageable parts if there are multiple errors."
            )},
                {"role": "user", "content": user_input}
            ],
        
    )
    print(response)
    return response.choices[0].message.content

# def get_languagetool_response(user_input):
#     matches = tool.check(user_input)
#     return [json.dumps(match.__dict__) for match in matches]

st.session_state.setdefault(
    'past', 
    []
)
st.session_state.setdefault(
    'generated', 
    []
)

def on_input_change():
    user_input = st.session_state.user_input
    language = st.session_state.lang_name
    print("bruh something happened", user_input, "fr")
    model_response = get_model_response(user_input, language)
    # grammar_response = get_languagetool_response(user_input)
    # print(grammar_response)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(model_response)
    # st.session_state.generated.append(str(grammar_response)) # right now just adding string version of grammar response


# def load_audio(file: tuple[str, bytes], sr: int = 16000):
#     """
#     Open an audio file and read as mono waveform, resampling as necessary

#     Parameters
#     ----------
#     file: (str, bytes)
#         The audio file to open or bytes of audio file

#     sr: int
#         The sample rate to resample the audio if necessary

#     Returns
#     -------
#     A NumPy array containing the audio waveform, in float32 dtype.
#     """
    
#     if isinstance(file, bytes):
#         inp = file
#         file = 'pipe:'
#     else:
#         inp = None
    
#     try:
#         # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
#         # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
#         out, _ = (
#             ffmpeg.input(file, threads=0)
#             .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
#             .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
#         )
#     except ffmpeg.Error as e:
#         raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# Function to convert audio to text w/ Whisper
def convert_audio_to_text(audio_data, language_code):
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
    #     tmp_file.write(audio_data)
    #     tmp_file_path = tmp_file.name

    # audio = whisper.load_audio(tmp_file_path)
    # audio = whisper.pad_or_trim(audio)
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)
    #mel = whisper.log_mel_spectrogram(load_audio(audio_data)).to(model.device)
    #options = whisper.DecodingOptions(language=language_code)
    #result = whisper.decode(model, mel, options)
    audio_file = io.BytesIO(audio_bytes)
    result = whisper.transcribe(audio_file, language=language_code)
    return result.text

def recording_callback():
    if st.session_state.my_recorder_output:
        language_code = language = st.session_state.lang_code
        audio_bytes = st.session_state.my_recorder_output['bytes']
        audio_text = convert_audio_to_text(audio_bytes, language_code)
        st.session_state.user_input = audio_text
        on_input_change()
        st.rerun()

chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state['past'])):                
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(st.session_state['generated'][i], is_user=False, key=f"{i}")
    
    st.button("Clear message", on_click=on_btn_click)

with st.container():
    # Main app logic
    st.title("Language Chat")
    mic_recorder(key='my_recorder', callback=recording_callback)
    st.text_input("User Input:", on_change=on_input_change, key="user_input")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("What is the language code?", value="es", key="lang_code")
    with c2:
        st.text_input("What is the language name?", value="Spanish", key="lang_name")
