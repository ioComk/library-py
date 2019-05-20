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
    
    # 信号の両端を零詰め
    for i in range(int(frame/2)):
        signal = np.insert(signal,i,0)
    for i in range(frame):
        signal = np.append(signal,0)

    # 窓関数の指定
    if windowtype == 'hann':
        window = np.hanning(frame)
    elif windowtype == 'hamming':
        window = np.hamming(frame)
    elif windowtype == 'blackman':
        window = np.blackman(frame)

    # STFT計算部分
    while frame <= len(y):
        try:
            # 窓関数を乗じてDFT
            mat[:, col] = abs(fft(y[begin:frame])).T*window
        # 例外処理
        except IndexError:
            print('IndexError. Frame size is too short.')
            break
            
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

    spectrogram = stft(y, 1000, 150,'hamming')

    plt.imshow(spectrogram)
    plt.colorbar()
    plt.show()