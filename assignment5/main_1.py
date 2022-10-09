import os
import pickle
import random
from typing import List
from matplotlib import pyplot as plt

import numpy as np
from scipy.io import wavfile

from PattRecClasses import FeatureExtractor, GetMusicFeatures, HMM, MarkovChain, multiGaussD

N = 4  # state number
TN = 10  # training number
TS = 2  # testing number


def normalize(xs: List[float]) -> List[float]:
    xs_sum = sum(xs)
    return [x / xs_sum for x in xs]


def findmin(x):
    a = len(x[0])
    for i in x:
        if len(i) < a:
            a = len(i)
    return a


def trans(x, minimum):
    for i in range(0, len(x)):
        x[i] = x[i][0:minimum]
        x[i] = np.array(x[i])
    return x


def reshapeobs(x):
    obs = []
    for i in x:
        obs.append(i.reshape(-1, 1))
    obs = np.array(obs)
    return obs


def readfeature(song, mydir, Train=True):
    sample_rate = []
    sig = []
    sample = []
    feature = []
    count = -1
    for i in song:
        count += 1
        if count <= TN - 1 and Train:
            rate_temp, sig_temp = wavfile.read(os.path.join(mydir, i))
            sample_rate.append(rate_temp)
            sig.append(sig_temp)
        if count > TN - 1 and Train == False:
            rate_temp, sig_temp = wavfile.read(os.path.join(mydir, i))
            sample_rate.append(rate_temp)
            sig.append(sig_temp)

    for j in range(0, TN if Train else TS):
        sample.append(GetMusicFeatures.GetMusicFeatures(sig[j], sample_rate[j]))
    for k in sample:
        feature.append(FeatureExtractor.FeatureExtractor(k))
    return feature


def testing(hmm_train, feature):
    for k in range(0, TS):
        exec("temp%s=[]" % k)
    for i in range(0, 3):
        for j in range(0, TS):
            a, b, c = hmm_train[i].calcabc(np.array(feature[j]).reshape((-1, 1)))
            logP = np.sum(np.log(c))
            exec("temp%s.append(logP)" % j)
            print('logP of sample %s = %s in HMM%s' % (j + 1, logP, i))
    for i in range(0, TS):
        print('The sample%s is recognized as song' % i)
        exec("print((temp%s.index(max(temp%s))+1))" % (i, i))


def main():
    try:
        with open("training_result.pkl", 'rb') as para:
            hmm_train = pickle.load(para)
            print("Read pretrained HMM data")
    except FileNotFoundError:
        print("Reading new wave data")
        dir1 = './Songs/A'
        songA = os.listdir(dir1)
        songA.sort(key=lambda x: int(x[0:-4]))
        dir2 = './Songs/B'
        songB = os.listdir(dir2)
        songB.sort(key=lambda x: int(x[0:-4]))
        dir3 = './Songs/C'
        songC = os.listdir(dir3)
        songC.sort(key=lambda x: int(x[0:-4]))
        '''
        Extract Features for each melody
        '''
        print("Reading new feature data")
        feature1 = readfeature(songA, dir1)
        feature2 = readfeature(songB, dir2)
        feature3 = readfeature(songC, dir3)

        '''
        Training Process
        '''
        print("Training new HMM data")
        q = np.array(normalize([1 / N + random.uniform(-0.2, 0.2) for _ in range(N)]))
        A = np.array([np.array(normalize([1 / N + random.uniform(-0.1, 0.1) for _ in range(N)])) for _ in range(N)])
        means = np.array([[0], [0], [0], [0]])
        covs = np.array([[[1]], [[1]], [[1]], [[1]]])
        mc = MarkovChain.MarkovChain(q, A)
        B = np.array([multiGaussD.multiGaussD(means[0], covs[0]),
                      multiGaussD.multiGaussD(means[1], covs[1]),
                      multiGaussD.multiGaussD(means[2], covs[2]),
                      multiGaussD.multiGaussD(means[3], covs[3])])
        feature1 = trans(feature1, findmin(feature1))
        obs1 = reshapeobs(feature1)
        hmm1 = HMM.HMM(mc, B)
        hmm1.Baum_Welch(obs1, 5, prin=1, uselog=False)

        feature2 = trans(feature2, findmin(feature2))
        obs2 = reshapeobs(feature2)
        hmm2 = HMM.HMM(mc, B)
        hmm2.Baum_Welch(obs2, 5, prin=1, uselog=False)

        feature3 = trans(feature3, findmin(feature3))
        obs3 = reshapeobs(feature3)
        hmm3 = HMM.HMM(mc, B)
        hmm3.Baum_Welch(obs3, 5, prin=1, uselog=False)

        learn = np.array([hmm1, hmm2, hmm3])
        output = open("training_result.pkl", 'wb')
        pickle.dump(learn, output)
        output.close()

        with open("training_result.pkl", 'rb') as para:
            hmm_train = pickle.load(para)

    print("Reading test wave data")
    dir1 = './Songs/C'
    songB = os.listdir(dir1)
    songB.sort(key=lambda x: int(x[0:-4]))

    print("Reading test feature data")
    feature1 = readfeature(songB, dir1, Train=False)

    res = hmm_train[2].rand(252)
    res1 = res[0]

    plt.plot(feature1[0], label='training sequence')
    plt.plot(res1[:, 0], label='random output sequence')
    plt.legend()
    plt.savefig('res3')


if __name__ == "__main__":
    main()


