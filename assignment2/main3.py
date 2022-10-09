import wave
import pylab as pl
import numpy as np

# plot signal and zoom in
f1 = wave.open(r"songs/melody_1.wav", "rb")
params1 = f1.getparams()
nchannels1, sampwidth1, framerate1, nframes1 = params1[:4]
print(nframes1)
str_data1 = f1.readframes(nframes1)
f1.close()
time1 = np.arange(0, nframes1) * (1.0/framerate1)
wav_data1 = np.frombuffer(str_data1, dtype=np.short)
wav_data1 = np.reshape(wav_data1, [-1,1]).T
pl.plot(time1, wav_data1[0],label='melody_1')

f2 = wave.open(r"songs/melody_2.wav", "rb")
params2 = f2.getparams()
nchannels2, sampwidth2, framerate2, nframes2 = params2[:4]
print(nframes2)
str_data2 = f2.readframes(nframes2)
f2.close()
time2 = np.arange(0, nframes2) * (1.0/framerate2)
wav_data2 = np.frombuffer(str_data2, dtype=np.short)
wav_data2 = np.reshape(wav_data2, [-1,1]).T
pl.plot(time2, wav_data2[0],label='melody_2')

f3 = wave.open(r"songs/melody_3.wav", "rb")
params3 = f3.getparams()
nchannels3, sampwidth3, framerate3, nframes3 = params3[:4]
print(nframes3)
str_data3 = f3.readframes(nframes3)
f3.close()
time3 = np.arange(0, nframes3) * (1.0/framerate3)
wav_data3 = np.frombuffer(str_data3, dtype=np.short)
wav_data3 = np.reshape(wav_data3, [-1,1]).T
pl.plot(time3, wav_data3[0],label='melody_3')
pl.xlabel("time(seconds)")
pl.ylabel("amplitude")
pl.legend()
pl.show()


