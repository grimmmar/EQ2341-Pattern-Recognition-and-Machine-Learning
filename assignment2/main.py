from matplotlib import pyplot as plt
from scipy.io import wavfile
import GetMusicFeatures
import FeatureExtractor
import random


# read melodies
sr1,wd1=wavfile.read('Songs/melody_1.wav')
sr2,wd2=wavfile.read('Songs/melody_2.wav')
sr3,wd3=wavfile.read('Songs/melody_3.wav')

frIsequence1=GetMusicFeatures.GetMusicFeatures(wd1,sr1) # default winlength=0.03
frIsequence2=GetMusicFeatures.GetMusicFeatures(wd2,sr2)
frIsequence3=GetMusicFeatures.GetMusicFeatures(wd3,sr3)

time1=list(range(len(frIsequence1[0,:])))
time2=list(range(len(frIsequence2[0,:])))
time3=list(range(len(frIsequence3[0,:])))

#pitch
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(time1,frIsequence1[0,:],color='black',linewidth=2.0)
plt.ylabel('Frequency/Hz')
plt.title('melody 1')

plt.subplot(3,1,2)
plt.plot(time2,frIsequence2[0,:],color='black',linewidth=2.0)
plt.ylabel('Frequency/Hz')
plt.title('melody 2')

plt.subplot(3,1,3)
plt.plot(time3,frIsequence3[0,:],color='black',linewidth=2.0)
plt.xlabel('Time frame')
plt.ylabel('Frequency/Hz')
plt.tight_layout()
plt.title('melody 3')
plt.savefig('pitch.png', dpi=600)

# intensity
plt.figure(2)
plt.subplot(3,1,1)
plt.plot(time1,frIsequence1[2,:],color='red',linewidth=2.0)
plt.ylabel('intensity')
plt.title('melody 1')

plt.subplot(3,1,2)
plt.plot(time2,frIsequence2[2,:],color='red',linewidth=2.0)
plt.ylabel('intensity')
plt.title('melody 2')

plt.subplot(3,1,3)
plt.plot(time3,frIsequence3[2,:],color='red',linewidth=2.0)
plt.xlabel('Time frame')
plt.ylabel('intensity')
plt.tight_layout()
plt.title('melody 3')
plt.savefig('intensity.png', dpi=600)

# correlation coefficient
plt.figure(3)
plt.subplot(1,3,1)
plt.plot(time1,frIsequence1[1,:],color='blue',linewidth=2.0)
plt.xlabel('Time Frame')
plt.ylabel('Correlation Coefficient')
plt.title('melody 1')

plt.subplot(1,3,2)
plt.plot(time2,frIsequence2[1,:],color='blue',linewidth=2.0)
plt.xlabel('Time Frame')
plt.title('melody 2')

plt.subplot(1,3,3)
plt.plot(time3,frIsequence3[1,:],color='blue',linewidth=2.0)
plt.xlabel('Time frame')
plt.title('melody 3')
plt.tight_layout()
plt.savefig('Correlation Coefficient.png', dpi=600)

#feature extract results
pitch1=FeatureExtractor.ChoosePitch(frIsequence1)
pitch1_n=FeatureExtractor.AddNoise(pitch1)
feature1=FeatureExtractor.Semitone(pitch1)

pitch2=FeatureExtractor.ChoosePitch(frIsequence2)
pitch2_n=FeatureExtractor.AddNoise(pitch2)
feature2=FeatureExtractor.Semitone(pitch2_n)

pitch3=FeatureExtractor.ChoosePitch(frIsequence3)
pitch3_n=FeatureExtractor.AddNoise(pitch3)
feature3=FeatureExtractor.Semitone(pitch3_n)

plt.figure(4)
plt.plot(time1,feature1)
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.title('Feature of melody 1')
plt.savefig('feature1.png', dpi=600)

plt.figure(5)
plt.plot(time2,feature2)
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.title('Feature of melody 2')
plt.savefig('feature2.png', dpi=600)

plt.figure(6)
plt.plot(time3,feature3)
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.title('Feature of melody 3')
plt.savefig('feature3.png', dpi=600)


#different transposition
frIsequence1_dt=GetMusicFeatures.GetMusicFeatures(wd1,sr1)
for i in range(0,len(frIsequence1_dt[0,:])):
    frIsequence1_dt[0,i]=1.5*frIsequence1_dt[0,i]
pitch1_dt=FeatureExtractor.ChoosePitch(frIsequence1_dt)
pitch1_dt_n=FeatureExtractor.AddNoise(pitch1_dt)
feature1_dt=FeatureExtractor.Semitone(pitch1_dt_n)

frIsequence2_dt=GetMusicFeatures.GetMusicFeatures(wd2,sr2)
for i in range(0,len(frIsequence2_dt[0,:])):
    frIsequence2_dt[0,i]=1.5*frIsequence2_dt[0,i]
pitch2_dt=FeatureExtractor.ChoosePitch(frIsequence2_dt)
pitch2_dt_n=FeatureExtractor.AddNoise(pitch2_dt)
feature2_dt=FeatureExtractor.Semitone(pitch2_dt_n)

plt.figure(7)
plt.plot(time1,feature1_dt,label='melody 1 with transposition')
plt.plot(time1,feature1,label='melody 1')
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.legend()
plt.title('Feature of melody 1 with 1.5 transposition')
plt.savefig('feature1_Trans.png', dpi=600)

plt.figure(8)
plt.plot(time2,feature2_dt,label='melody 2 with transposition')
plt.plot(time2,feature2,label='melody 2')
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.legend()
plt.title('Feature of melody 2 with 1.5 transposition')
plt.savefig('feature2_Trans.png', dpi=600)


#different volume
wd4=wd1.copy()
for i in range(0,len(wd4)):
    wd4[i]=2*wd4[i]
frIsequence_dv1=GetMusicFeatures.GetMusicFeatures(wd4,sr1)
pitch_dv1=FeatureExtractor.ChoosePitch(frIsequence_dv1)
pitch_dv1_n=FeatureExtractor.AddNoise(pitch_dv1)
feature_dv1=FeatureExtractor.Semitone(pitch_dv1_n)


wd5=wd1.copy()
for i in range(0,len(wd5)):
    wd5[i]=0.5*wd5[i]
frIsequence_dv2=GetMusicFeatures.GetMusicFeatures(wd5,sr1)
pitch_dv2=FeatureExtractor.ChoosePitch(frIsequence_dv2)
pitch_dv2_n=FeatureExtractor.AddNoise(pitch_dv2)
feature_dv2=FeatureExtractor.Semitone(pitch_dv2_n)

plt.figure(9)
plt.plot(time1,feature_dv1,label='melody 1 with 2 volume')
plt.plot(time1,feature1,label='melody 1')
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.legend()
plt.title('Feature of melody 1 with 2 volume')
plt.savefig('feature1_highVolume.png', dpi=600)

plt.figure(10)
plt.plot(time1,feature_dv2,label='melody 1 with 0.5 volume')
plt.plot(time1,feature1,label='melody 1')
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.legend()
plt.title('Feature of melody 1 with 0.5 volume')
plt.savefig('feature1_lowVolume.png', dpi=600)

# brief episode
wd6=wd1.copy()
r1=random.sample(range(0,len(wd6)),5)
r2=random.sample(range(0,len(wd6)),5)
frIsequence_pj=GetMusicFeatures.GetMusicFeatures(wd6,sr1)
for i in range(0,len(wd6)):
    if i==r1:
        frIsequence_pj[0,i]=2*frIsequence_pj[0,i]
    if i==r2:
        frIsequence_pj[0,i]=0.5*frIsequence_pj[0,i]
pitch_pj=FeatureExtractor.ChoosePitch(frIsequence_pj)
pitch_pj_n=FeatureExtractor.AddNoise(pitch_pj)
feature_pj=FeatureExtractor.Semitone(pitch_pj_n)

plt.figure(11)
plt.plot(time1,feature_pj,label='melody 1 with pitch jump')
plt.plot(time1,feature1,label='melody 1')
plt.xlabel('Time frame')
plt.ylabel('Feature')
plt.legend()
plt.title('Feature of melody 1 with 10 pitch jump')
plt.savefig('feature1_pitch_jump.png', dpi=600)
