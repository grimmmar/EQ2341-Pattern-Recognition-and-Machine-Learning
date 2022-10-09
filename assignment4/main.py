from PattRecClasses import DiscreteD, GaussD, HMM, MarkovChain
import numpy as np
import math

# finite
q = np.array([1, 0])
A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
X = np.array([[-0.2, 2.6, 1.3]])

mc = MarkovChain(q, A)
g1 = GaussD(means=[0],stdevs=[1])   # Distribution for state = 1
g2 = GaussD(means=[3],stdevs=[2])   # Distribution for state = 2
h = HMM(mc, [g1, g2])
pX1 = GaussD.logprob([g1, g2], X)
for j in range(0, 3):
    for i in range(0, 2):
        pX1[i, j] = np.exp(pX1[i, j])
    temp = max(pX1[:, j])
    for i in range(0,2):
        pX1[i, j] = pX1[i, j]/temp
alfaHat, ct = mc.forward(pX1)
betaHat=mc.backward(ct, pX1)

print(ct)
print(betaHat)

# infinite
q2 = np.array([1, 0, 0])
A2 = np.array([[0.3, 0.7, 0], [0, 0.5, 0.5], [0, 0, 1]])
X2 = [1, 2, 4, 4, 1]

mc2 = MarkovChain(q2, A2)
b1 = DiscreteD([1, 0, 0, 0])
b2 = DiscreteD([0, 0.5, 0.4, 0.1])
b3 = DiscreteD([0.1, 0.1, 0.2, 0.6])
B = [b1, b2, b3]
pX21 = b1.prob(X2)
pX22 = b2.prob(X2)
pX23 = b3.prob(X2)
pX2 = np.concatenate((pX21, pX22, pX23))
pX2 = pX2.reshape(3, 5)
alfaHat2, ct2 = mc2.forward(pX2)
betaHat2 = mc2.backward(ct2, pX2)

print(betaHat2)
