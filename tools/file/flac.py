import wave
import numpy as np
import soundfile as sf

def save_flac(frames: np.ndarray, fname, sample_rate=44100):
    shape = list(frames.shape)
    if(len(shape) == 1):
        frames = frames[...,None]
    in_samples, in_channels = shape[-2], shape[-1]
    if(in_channels >= 3):
        if(len(shape) == 2):
            frames = np.transpose(frames,(1,0))
        elif(len(shape) == 3):
            frames = np.transpose(frames, (0,2,1))
        msg = "Warning: Save audio with "+str(in_channels) +" channels, save permute audio with shape "+str(list(frames.shape))+" please check if it's correct."
        # print(msg)
    if(np.max(frames) <= 1 and frames.dtype == np.float32 or frames.dtype == np.float16 or frames.dtype == np.float64):
        frames *= 2**15
    frames = frames.astype(np.short)
    if (len(frames.shape) >= 3):
        frames = frames[0,...]
    sf.write(fname,frames,samplerate=sample_rate, format='FLAC')
