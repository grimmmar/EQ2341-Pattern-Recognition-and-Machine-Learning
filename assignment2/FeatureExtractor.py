# -*- coding: utf-8 -*-
import numpy as np
import math
import GetMusicFeatures

def ChoosePitch(frIsequence):
    pitch=frIsequence[0,:]
    correlation=frIsequence[1,:]
    intensity=frIsequence[2,:]
    N=len(pitch)
    intensityLog=np.zeros(N)
    for i in range(0,N):
        intensityLog[i]=math.log(intensity[i])
    intensityNor=np.zeros(N)
    for i in range(0,N):
        intensityNor[i]=(intensityLog[i]-np.min(intensityLog))/(np.max(intensityLog)-np.min(intensityLog))
    thresholdPitch=np.mean(pitch)+np.std(pitch)
    thresholdCorr=0.75
    thresholdInten=np.mean(intensityNor)
    for i in range(0,N):
        if pitch[i]>thresholdPitch or correlation[i]<thresholdCorr or intensityNor[i]<thresholdInten:
            pitch[i]=0
    return pitch


def AddNoise(pitch):
    threshold=np.mean(pitch)-np.std(pitch)
    N=len(pitch)
    basicF=np.max(pitch)
    for i in range(0,N):
        if(pitch[i]>threshold):
            if(pitch[i]<basicF):
                basicF=pitch[i]
    for i in range(0,N):
        if(pitch[i]==0):
            pitch[i]=basicF+2*np.random.rand()
    return pitch
    
    
    
def Semitone(pitch):
    threshold=np.mean(pitch)-np.std(pitch)
    N=len(pitch)
    temp=np.max(pitch)
    for i in range(0,N):
        if(pitch[i]>threshold):
            if(pitch[i]<temp):
                temp=pitch[i]
    semi=np.zeros(N)
    for i in range(0,N):
        semi[i]=12*math.log(pitch[i]/temp,2)
    return semi
    

