'''
frIsequence = GetMusicFeatures(signal,fs)
or
frIsequence = GetMusicFeatures(signal,fs,winlength)

Method to calculate features for melody recognition

Usage:
First load a sound file using wavfile.read(...) or similar, then use this function
to extract pitch and energy contours of the melody in the sound. This
information can be used to compute a sequence of feature values or
vectors for melody recognition. Note that the pitch estimation is
unreliable (typically giving very high values) in silent segments, and
may not work at all for polyphonic sounds.

Input:
signal=      Vector containing sampled signal values (must be mono).
fs=          Sampling frequency of signal in Hz.
winlength=   Length of the analysis window in seconds (default 0.03).
             Square ("boxcar") analysis windows with 50% overlap are used.

Output:
frIsequence= Matrix containing pitch, correlation, and intensity estimates
             for use in creating features for melody recognition. Each column
             represents one frame in the analysis. Elements in the first
             row are pitch estimates in Hz (80--1100 Hz), the second row
             estimates the correlation coefficient (rho) between adjacent
             pitch periods, while the third row contains corresponding
             estimates of per-sample intensity.

References:
This method is based on a pitch estimator provided by Obada Alhaj Moussa.
'''
import numpy as np

def GetMusicFeatures(signal, fs, winlength=0.03):

    # Wikipedia: "human voices are roughly in the range of 80 Hz to 1100 Hz"
    minpitch = 80
    maxpitch = 1100

    signal = np.real(np.double(signal)) # Make sure the signal is a real double

    signal = signal - np.mean(signal) # Remove DC, which can disturb intesities

    if fs <= 0:
        fs = samplerate # Replace illegal fs-values with the read sampling freq.

    # Compute the pitch periods in samples for the human voice range
    minlag = int( np.round(fs/maxpitch) )
    maxlag = int( np.round(fs/minpitch) )

    winlength = np.abs(winlength)
    winlength = np.round(winlength*fs) # Convert to number of samples
    winlength = max([ winlength+(winlength % 2), 2*minlag ]) # Make windows sufficiently long and an even sample number

    winstep = int( winlength/2 );
    nsteps = int( np.floor(len(signal)/winstep) ) - 1

    if (nsteps < 1):
        print(['ERROR: Signal too short. Use at least %s samples!'%(str(winlength))])
        return None

    frIsequence = np.zeros((3,nsteps)) # Initialize output variable to correct size

    for n in range(nsteps):
        # Cut out a segment of the signal starting at offset n*winlength sec
        window = signal[n*winstep : (n+1)*winstep]

        # Estimate the pitch (sampling frequency/pitch period), between-period
        # correlation coefficient, and intensity
        pprd, maxcorr = yin_pitch(window,minlag,maxlag)
        frIsequence[:,n] = [fs/pprd, maxcorr, np.linalg.norm(window/np.sqrt(len(window)))]
        
    return frIsequence

'''
Below is the pitch period estimation sub-routine.
The estimate is based on the autocorrelation function.
'''
def yin_pitch(signal,minlag=40,maxlag=200):

    N = len(signal)
    
    dif = np.zeros(maxlag - minlag)
    for idx in range(minlag, maxlag):
        seg1 = signal[idx : ]
        seg2 = signal[ : N - idx]

        # Estimate correlation ("dif") at lag idx
        dif[idx - minlag] = sum((seg1 - seg2)**2) / (N - idx)

    thresh = (max(dif) - min(dif)) * 0.1 + min(dif);

    
    # Locate the first minimum of dif, which is the first maximum of the
    # correlation; the corresponding lag is the pitch period.
    pprd = None
    idx = minlag
    while ( idx < maxlag ):
        if dif[idx - minlag] <= thresh:
            pprd = idx
            break

        idx = idx + 1

    # Allow the procedure to find the first minimum to roll over small "bumps"
    # in the autocorrelation functions, that are below than a 10% threshold.
    while idx < maxlag:
        if dif[idx - minlag] >= thresh:
            break

        if dif[idx - minlag] < dif[pprd - minlag]:
            pprd = idx

        idx = idx + 1

    seg1 = signal[pprd : ]
    seg2 = signal[: N - pprd]

    maxcorr = np.corrcoef(seg1,seg2)[0,1]
    
    return (pprd, maxcorr)