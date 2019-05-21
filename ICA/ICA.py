#!/usr/bin/python
import numpy as np
import soundfile as sf
import random as rnd
import math as m
import multiprocessing as mp


# スコア関数(今はtanhのみ)
def score(x):
    return m.tanh(x)

# 部分和計算
def calc_partial_sum(dataset):
    wavs, begin, end = dataset
    entropy = np.zeros([2, 2])
    for t in range(begin, end):
        orig = np.zeros([1, len(wavs)])
        scored = np.zeros([len(wavs), 1])
        for i in range(0, len(wavs)):
            orig[0][i] = wavs[i][t]
            scored[i][0] = score(wavs[i][t])
        phi = scored @ orig
        entropy = entropy + phi
    return entropy

def entropy(wavs):
    cpu_num = mp.cpu_count()
    step = int(m.ceil(len(wavs[0]) / cpu_num))
    pool = mp.Pool(cpu_num)

    cfgs = []
    for i in range(0, cpu_num):
        cfgs.append((wavs, i * step, min((i+1) * step, len(wavs[0]) - 1)))

    entropy = np.zeros([2, 2])
    for sub in pool.map(calc_partial_sum, cfgs):
        entropy = entropy + sub
    pool.close()
    return entropy / len(wavs[0])

def ICA(wavs, iter=100, step=0.2):
    ipt = np.array(wavs)
    w = np.array([[rnd.random() - 0.5, rnd.random() - 0.5], [rnd.random() - 0.5, rnd.random() - 0.5]])
    out = w @ ipt
    for i in range(0, iter):
        e = entropy(out)
        w = w + step * (np.identity(2) - e) @ w
        out = w @ ipt
    return w

if __name__ == '__main__':
    wav1, rate1 = sf.read('./wav/mix1.wav')
    wav2, rate2 = sf.read('./wav/mix2.wav')

    w = ICA([wav1, wav2])
    out = w @ np.array([wav1, wav2])

    sf.write('ica1.wav', out[0], rate1)
    sf.write('ica2.wav', out[1], rate2)
