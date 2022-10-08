import numpy as np
import matplotlib.pyplot as plt
from module import DiscreteD, GaussD, MarkovChain, HMM

#2
T=10000
n=1
count = np.zeros(n)
for j in range(n):
    mc_1 = MarkovChain(np.array([0.75, 0.25]), np.array([[0.99, 0.01], [0.03, 0.97]]))
    S = mc_1.rand(T)
    for i in range(0, T):
        if S[i] == 1:
            count[j] += 1
avg1=np.mean(count)
print(avg1)

#3
m=1
mean=np.zeros(m)
var=np.zeros(m)
for i in range(m):
    mc_2 = MarkovChain(np.array([0.75, 0.25]), np.array([[0.99, 0.01], [0.03, 0.97]]))
    g1=GaussD(means=[0],stdevs=[1])   # Distribution for state = 1
    g2=GaussD(means=[3],stdevs=[2])   # Distribution for state = 2
    h=HMM(mc_2,[g1,g2])
    x,s=h.rand(T)
    mean[i]=np.mean(x)
    var[i]=np.var(x)
mean_avg=np.mean(mean)
var_avg=np.mean(var)
print(mean_avg,var_avg)

#4
mc = MarkovChain(np.array([0.75, 0.25]), np.array([[0.99, 0.01], [0.03, 0.97]]))
g1=GaussD(means=[0],stdevs=[1])   # Distribution for state = 1
g2=GaussD(means=[3],stdevs=[2])   # Distribution for state = 2
h=HMM(mc,[g1,g2])
x2,s2=h.rand(500)
sum1=np.zeros(500)
sum2=np.zeros(500)
for i in range(500):
    if s2[i]==1:
        sum1[i]=x2[0,i]
    if s2[i]==2:
        sum2[i]=x2[0,i]
plt.figure(1)
plt.plot(sum1,color='blue',label='St=1')
plt.plot(sum2,color='orange',label='St=2')
sum1=sum1[sum1!=0]
sum2=sum2[sum2!=0]
savg1=np.mean(sum1)
savg2=np.mean(sum2)
print(savg1,savg2)
plt.axhline(y=savg1,color='red',ls='--',label='mean of St=1')
plt.axhline(y=savg2,color='green',ls='--',label='mean of St=2')
plt.legend()
plt.xlabel('t')
plt.ylabel('HMM output X(t)')
plt.title('HMM output with 500 samples')
plt.savefig('4.png', dpi=600)

#5
g3=GaussD(means=[0],stdevs=[1])   # Distribution for state = 1
g4=GaussD(means=[0],stdevs=[2])   # Distribution for state = 2
h2=HMM(mc,[g3,g4])
x3,s3=h2.rand(500)
sum3=np.zeros(500)
sum4=np.zeros(500)
for i in range(500):
    if s3[i]==1:
        sum3[i]=x3[0,i]
    if s3[i]==2:
        sum4[i]=x3[0,i]
plt.figure(2)
plt.plot(sum3,color='blue',label='St=1')
plt.plot(sum4,color='orange',label='St=2')
sum3=sum3[sum3!=0]
sum4=sum4[sum4!=0]
savg3=np.mean(sum3)
savg4=np.mean(sum4)
print(savg3,savg4)
plt.axhline(y=savg3,color='red',ls='--',label='mean of St=1')
plt.axhline(y=savg4,color='green',ls='--',label='mean of St=2')
plt.legend()
plt.xlabel('t')
plt.ylabel('HMM output X(t)')
plt.title('HMM output with 500 samples')
plt.savefig('5.png', dpi=600)

#6
mcd=MarkovChain(np.array([0.75,0.25]),np.array([[0.4,0.5,0.1],[0.7,0.3,0.1]]))
hd=HMM(mcd,[g1,g2])
above=np.zeros(1000)
for m1 in range(0,1000):
    xd,sd=hd.rand(1000)
    temp=0
    for m2 in range(0,1000):
        if sd[m2]==1:
            temp=temp+1
        if sd[m2]==2:
            temp=temp+1
    above[m1]=temp
above_max=max(above)
above_min=min(above)
average=np.mean(above)
print(above_max,above_min,average)

#7
g5=GaussD(means=np.array([0,1]),stdevs=np.array([1,2]))
g6=GaussD(means=np.array([1,2]),stdevs=np.array([2,4]))
h4=HMM(mc,[g5,g6])
x4,s4=h4.rand(1000)
C=np.cov(x4[:,0],x4[:,1])
print(C)