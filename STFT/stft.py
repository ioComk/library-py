import os
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def stft(signal, frame, shift, windowtype='hann'):
    begin = 0
    col = 0  # 配列に格納する際の列のインデックス

    mat = np.zeros((frame, int(len(y)/shift)))  # スペクトログラムを格納する行列，大体の大きさを確保しておく
    if windowtype == 'hann':
        window = np.hanning(frame)
    elif windowtype == 'hamming':
        window = np.hamming(frame)
    elif windowtype == 'blackman':
        window = np.blackman(frame)

    while frame <= len(y):
        mat[:, col] = abs(fft(y[begin:frame])).T*window
        # それぞれシフト長だけシフト
        begin += shift
        frame += shift
        col += 1

    mat = mat[:, :col]  # 余分な列を削除
    # 下半分だけ抽出して利得を計算
    mat = 20*np.log1p(mat[int(mat.shape[0]/2):int(mat.shape[0]), :])

    return mat

if __name__ == '__main__':
    filename = 'wav/s1.wav'

    # y = amplitude, sr = sample rate
    y, sr = sf.read(filename)

    spectrogram = stft(y, 1000, 150, 'hamming')

    plt.imshow(spectrogram, cmap="inferno")
    plt.colorbar()
    plt.show()