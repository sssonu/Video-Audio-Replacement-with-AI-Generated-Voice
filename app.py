import streamlit as st
import moviepy.editor as mp
import torch
import os
from transformers import pipeline
from TTS.api import TTS  # Coqui TTS
import requests  # For LanguageTool API

# Title and description
st.title("Video Audio Replacement with AI Generated Voice")
st.write("This PoC takes a video file, transcribes its audio, corrects grammar, and replaces the audio with AI-generated voice.")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Ensure ffmpeg is available
os.system("ffmpeg -version")

if uploaded_file is not None:
    # Save uploaded video
    video_path = uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract audio from the video
    video = mp.VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)

    st.audio(audio_path)

    # Step 2: Speech to Text using Whisper (free, local model)
    st.write("Transcribing the audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if device == "cuda" else -1)
    
    # Transcribing the audio
    transcription = transcriber(audio_path)["text"]
    st.write("Transcription:")
    st.write(transcription)

    # Step 3: Grammar correction using LanguageTool API
    st.write("Correcting transcription grammar...")

    def correct_grammar_with_api(text):
        url = "https://api.languagetool.org/v2/check"
        data = {
            "text": text,
            "language": "en-US"
        }
        response = requests.post(url, data=data)
        result = response.json()
        corrected_text = text
        for match in reversed(result['matches']):
            start = match['offset']
            end = start + match['length']
            replacement = match['replacements'][0]['value'] if match['replacements'] else ""
            corrected_text = corrected_text[:start] + replacement + corrected_text[end:]
        return corrected_text

    corrected_transcription = correct_grammar_with_api(transcription)
    
    st.write("Corrected Transcription:")
    st.write(corrected_transcription)

    # Step 4: Text-to-Speech using Coqui TTS (with a different model)
    st.write("Generating new audio with corrected transcription...")

    # Change to another available model (new model)
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
    
    # Save the newly generated audio
    new_audio_path = "corrected_audio.wav"
    tts_model.tts_to_file(text=corrected_transcription, file_path=new_audio_path)

    st.audio(new_audio_path)

    # Step 5: Replace old audio with new audio in the video
    st.write("Replacing the audio in the video...")
    final_video_path = "output_video_with_corrected_audio.mp4"
    
    # Replace audio in the original video
    final_video = video.set_audio(mp.AudioFileClip(new_audio_path))
    final_video.write_videofile(final_video_path)

    # Play final video with corrected audio
    st.video(final_video_path)

    st.success("Process completed. Download the final video below:")
    with open(final_video_path, "rb") as f:
        st.download_button("Download Final Video", f, file_name="output_video_with_corrected_audio.mp4")
