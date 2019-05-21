import os
import soundfile as sf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def stft(signal, fftsize, shiftsize, windowtype='hann'):
    begin = 0
    col = 0  # 配列に格納する際の列のインデックス

    mat = np.zeros((fftsize, (len(signal)+int(fftsize*1.5))//shiftsize))  # スペクトログラムを格納する行列，大体の大きさを確保しておく

    # 信号の両端を零詰め
    zero_padding = np.zeros((fftsize))
    signal = np.concatenate([zero_padding[0:fftsize//2], signal, zero_padding])

    # 窓関数の指定
    if windowtype == 'hann':
        window = np.hanning(fftsize)
    elif windowtype == 'hamming':
        window = np.hamming(fftsize)
    elif windowtype == 'blackman':
        window = np.blackman(fftsize)

    # STFT計算部分
    while fftsize <= len(signal):
        try:
            # 窓関数を乗じてDFT
            mat[:, col] = abs(fft(signal[begin:fftsize])).T*window
        # 例外処理
        except IndexError:
            print('IndexError. FFT size is too short.')
            break

        # それぞれシフト長だけシフト
        begin += shiftsize
        fftsize += shiftsize
        col += 1

    mat = mat[:, :col]  # 余分な列を削除
    # 下半分だけ抽出して利得を計算
    mat = 20*np.log1p(mat[mat.shape[0]//2:mat.shape[0], :])

    return mat

if __name__ == '__main__':
    filename = 'wav/s1.wav'

    # y = amplitude, sr = sample rate
    y, sr = sf.read(filename)

    spectrogram = stft(y, 1000, 150, 'hamming')

    plt.imshow(spectrogram)
    plt.colorbar()
    plt.show()