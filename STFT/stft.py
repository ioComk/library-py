import os
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def stft(signal, frame, shift):
    begin = 0
    col = 0  # 配列に格納する際の列のインデックス

    mat = np.zeros((frame, int(len(y)/shift)))  # スペクトログラムを格納する行列，大体の大きさを確保しておく

    while frame <= len(y):
        mat[:, col] = abs(fft(y[begin:frame])).T
        # それぞれシフト長だけシフト
        begin += shift
        frame += shift
        col += 1

    mat = mat[:, :col]  # 余分な列を削除
    mat = 20*np.log1p(mat[int(mat.shape[0]/2):int(mat.shape[0]), :])  # 下半分だけ抽出して利得を計算
    
    return mat

if __name__ == '__main__':
    filename = 'wav/s1.wav'

    # y = amplitude, sr = sample rate
    y, sr = sf.read(filename)

    spectrogram = stft(y,1000,150)

    plt.imshow(spectrogram, cmap="inferno")
    plt.colorbar()
    plt.show()