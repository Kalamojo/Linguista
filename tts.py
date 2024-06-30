import streamlit as st
import whisper
import language_tool_python
import io
import ffmpeg
import numpy as np
#import torch

# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai_key)
model = whisper.load_model('base')

tool = language_tool_python.LanguageTool('ja')

def load_audio(file: tuple[str, bytes], sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None
    
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def recording_callback():
    if st.session_state.my_recorder_output:
        # device = torch.device("cuda")
        # model.cuda()
        audio_bytes = st.session_state.my_recorder_output['bytes']
        # audio_bio = io.BytesIO(audio_bytes)
        # audio_bio.name = 'audio.webm'
        print('doin whisper stuff')
        mel = whisper.log_mel_spectrogram(load_audio(audio_bytes)).to('cuda')
        options = whisper.DecodingOptions(language="en")
        print(mel)
        transcription = whisper.decode(model, mel, options)
        # transcription = client.audio.transcriptions.create(
        #     model="whisper-1",
        #     file=audio_bio,
        #     language="ja"
        # )
        st.write(transcription.text)
        matches = tool.check(transcription.text)
        print(matches)
        st.write(matches)
