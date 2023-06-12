import argparse
from models.gsr_voicefixer import VoiceFixer
from models.gsr_voicefixer_production import VoiceFixer as VoiceFixerProduction
import torch

@torch.no_grad()
def main(input_file, output_file, checkpoint, **kwargs):
    model = VoiceFixer.load_from_checkpoint(checkpoint)
    model = VoiceFixerProduction(model) 
    print(model)
    model.restore(input=input_file, output=output_file, cuda=True, mode=0)       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Processing Script")
    parser.add_argument("--input_file", help="Path to the input audio file (FLAC or WAV)", required=True)
    parser.add_argument("--output_file", help="Path to the output audio file", required=True)
    parser.add_argument("--checkpoint", help="Path to the model checkpoint file", required=True)

    args = parser.parse_args()

    main(**vars(args))