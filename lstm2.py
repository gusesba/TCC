import librosa
import numpy as np

y, sr = librosa.load("./segment_3.wav", sr=2000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  
mfcc = mfcc.T  # agora fica shape (time_steps, 40)
print(mfcc.shape)