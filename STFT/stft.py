import os
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft


if __name__ == '__main__':
    filename = 'wav/s1.wav'

    # y = amplitude, sr = sample rate
    y, sr = sf.read(filename)

    begin = 0

    frame = 1000
    shift = 150  # シフト長
    col = 0

    totaltime = len(y)/sr
    mat = np.zeros((frame, int(len(y)/shift)))  # 作成するスペクトログラム
    
    while frame <= len(y):
        buf = abs(fft(y[begin:frame]))
        mat[:, col] = buf.T
        begin += shift
        frame += shift
        col += 1
    
    mat = mat[:, :col]  # 余分な列を削る
    
    # 下半分を抽出し利得を計算
    mat_map = 20*np.log1p(mat[int(mat.shape[0]/2):int(mat.shape[0]), :])
    
    plt.imshow(mat_map, cmap="inferno")
    plt.colorbar()
    plt.show()