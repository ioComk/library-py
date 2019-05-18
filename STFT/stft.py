import os
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def stft(signal, frame, shift):
    begin = 0
    col = 0
    mat = np.zeros((frame, int(len(y)/shift)))

    while frame <= len(y):
        mat[:, col] = abs(fft(y[begin:frame])).T
        begin += shift
        frame += shift
        col += 1

    mat = mat[:, :col]
    mat = 20*np.log1p(mat[int(mat.shape[0]/2):int(mat.shape[0]), :])
    
    return mat

if __name__ == '__main__':
    filename = 'wav/s1.wav'

    # y = amplitude, sr = sample rate
    y, sr = sf.read(filename)

    spectrogram = stft(y,1000,150)

    plt.imshow(spectrogram, cmap="inferno")
    plt.colorbar()
    plt.show()