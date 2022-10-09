'''
We will use the python_speech_features library for 
extracting the MFCC components for speech sounds.

You can find the installation instructions and other
ïnformation regarding this library here:

https://pypi.org/project/python_speech_features/0.4/

This library also supports additional speech features 
beyond MFCC. You are allowed to use such features if you
find them interesting/relevant for the project task. 
However, consider such features strictly optional and
an add-on to the MFCC. If you end up using more feaures,
please make sure to highlight them clearly in your final 
report and presentation. Good luck!
'''

from python_speech_fatures import mfcc

def GetSpeechFeatures(signal,fs):
    features_mfcc = mfcc(signal, samplerate)
    
    return features_mfcc
    
    