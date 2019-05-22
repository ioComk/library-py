import os
import sys
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft

'''
# args

      signal: input signal
     fftsize: frame length (Even number)
   shiftsize: shift length (Recommend: fftsize/2)
  windowtype: window function

# returns

       S: spectrogram of input signal
  window: window function used in STFT

'''


def stft(signal, fftsize=1024, shiftsize=512, windowtype='hamming'):

    if fftsize % 2 != 0:
        print('FFT size must be an even number.')
        sys.exit()
    elif fftsize % shiftsize != 0:
        print('FFT size must be dividable by Shift size.')
        sys.exit()

    # 窓関数の指定
    if windowtype == 'hann':
        window = np.hanning(fftsize)
    elif windowtype == 'hamming':
        window = np.hamming(fftsize)
    elif windowtype == 'blackman':
        window = np.blackman(fftsize)
    else:
        print('Unsupported window is requested.')
        sys.exit()

    # 信号の両端を零詰め
    nch = signal.ndim
    zero_padding = fftsize - shiftsize
    frames = (len(signal) - fftsize + shiftsize) // shiftsize
    i = fftsize//2+1

    # calculate STFT
    # monoral
    if nch == 1:
        signal = np.concatenate(
            [np.zeros(zero_padding), signal, np.zeros(fftsize)])
        S = np.zeros([i, frames], dtype=np.complex128)

        for j in range(frames):
            sp = j*shiftsize
            spectrum = fft(signal[sp: sp+fftsize]*window)
            S[:, j] = spectrum[:i]

        return S, window
    # stereo
    elif nch == 2:
        signal = np.concatenate(
            [np.zeros((zero_padding, nch)), signal, np.zeros((zero_padding, nch))])
        S = np.zeros([i, frames, nch], dtype=np.complex128)

        for ch in range(nch):
            for j in range(frames):
                sp = j*shiftsize
                spectrum = fft(signal[sp: sp+fftsize, ch]*window)
                S[:, j, ch] = spectrum[:i]

        return S, window


def power(S):
    return 20*np.log1p(abs(S))


if __name__ == '__main__':
    filename = 'wav/s1.wav'

    # y = amplitude, sr = sample rate
    y, sr = sf.read(filename)

    spectrogram, window = stft(y, 1024, 512, 'hamming')

    plt.imshow(power(spectrogram))
    plt.colorbar()
    plt.show()
