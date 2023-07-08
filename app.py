import torch
import os
import gradio as gr

from audiomaister import VoiceFixer
from audiomaister.models.gs_audiomaister import AudioMaister

USE_CUDA = torch.cuda.is_available()

def load_default_weights():
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    REPO_ID = "peterwilli/audio-maister"
    print(f"Loading standard model weight at {REPO_ID}")
    MODEL_FILE_NAME = "audiomaister_v1.ckpt"
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE_NAME)
    return checkpoint_path

def inference(input_file, **kwargs):
    checkpoint = load_default_weights()
    state = torch.load(checkpoint, map_location=torch.device('cuda' if USE_CUDA else 'cpu'))
    
    main_model = VoiceFixer(state['hparams'], 1, 'vocals')
    main_model.load_state_dict(state['weights'])
    
    inference_model = AudioMaister(main_model)
    inference_model.restore(input=input_file, output="out.wav", mode=0)
    
    if USE_CUDA:
        main_model.to('cuda')
        inference_model.to('cuda')

    return "out.wav"

made ="""<div style="text-align: center;">
  <p>Made with ❤ by Raaniel</p>"""

desc = """<div style="text-align: left;"> AudiomAIster is a fork of VoiceFixer 
that focuses on general purpose audio (where VoiceFixer focusses on voice alone, 
AudiomAIster will reduce noise while enhancing voice and sound effects, 
making it more suitable for talks and videos where more than just voices exist).
<br><br>
<a href="https://github.com/peterwilli/audio-maister" target="_blank" rel="noopener noreferrer">
See the model main repository</a>
<br></div>
"""

gr.Interface(
    fn=inference, 
    inputs=gr.Audio(type="filepath", source="upload", label = "Upload the audio that needs to be fixed!"),
    outputs=gr.Audio(type = "filepath", label = "Your fixed audio is going to show up below: "),
    title="🩺🎧 Fix all your podcast, video or live stream audio! 🎧🩺",
    description = desc,
    article = made,
    theme=gr.themes.Soft(primary_hue="purple",secondary_hue="violet", neutral_hue="neutral")
).launch()