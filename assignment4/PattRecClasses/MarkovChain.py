import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message
        """
        S = DiscreteD(self.q).rand(1)
        if self.is_finite:
            t = 1
            while t < tmax:
                state_tplus1 = DiscreteD(self.A[S[t-1],:]).rand(1)
                if state_tplus1[0] == self.nStates:
                    return S
                S = np.concatenate((S, state_tplus1))
                t += 1
            return S
        else:
            for t in np.arange(1,tmax):
                state_tplus1 = DiscreteD(self.A[S[t-1],:]).rand(1)
                S = np.concatenate((S, state_tplus1))
            return S
            """
        S = np.zeros(tmax)
        nS = self.nStates
        endstate = nS + 1

        p = DiscreteD(self.q)
        S[0] = p.rand(1)[0]
        end = 0
        i = 1
        #        if self.A.shape[0]==self.A.shape[1]:
        for i in range(1, tmax):
            temp = S[i - 1].astype(np.int64)
            transstate = DiscreteD(self.A[temp - 1, :])
            S[i] = transstate.rand(1)[0]
            if S[i] == endstate:
                S[i:end] = []
                break
        return S


    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self,pX):
        T = len(pX[0, :])
        N = self.nStates
        q = self.q
        A = self.A
        c = np.zeros(T)
        alpha = np.zeros((N, T))
        alfaHat = np.zeros((N, T))
        finitestate = 0
        if N != len(A[0, :]):
            finitestate = 1

        # Initialization
        for i in range(0, N):
            alpha[i, 0] = q[i] * pX[i, 0]
            c[0] = c[0] + alpha[i, 0]
        for i in range(0, N):
            alfaHat[i, 0] = alpha[i, 0] / c[0]

        # Forward Steps
        for t in range(1, T):
            for j in range(0, N):
                alpha[j, t] = pX[j, t] * (np.transpose(alfaHat[:, t - 1]) @ A[:, j])
                c[t] = c[t] + alpha[j, t]
            for j in range(0, N):
                alfaHat[j, t] = alpha[j, t] / c[t]

        for t in range(1, T):
            for j in range(0, N):
                alfaHat[j, t] = alpha[j, t] / c[t]

        # Termination
        if finitestate == 1:
            tempc = c
            c = np.zeros(T + 1)
            for i in range(0, T):
                c[i] = tempc[i]
            for i in range(0, N):
                c[T] = c[T] + alfaHat[i, T - 1] * A[i, N]

        return alfaHat, c

    def finiteDuration(self):
        pass
    
    def backward(self, c, pX):
        T = len(pX[0, :])
        N = self.nStates
        A = self.A
        betaHat = np.zeros((N, T))
        finitestate = 0
        if N != len(A[0, :]):
            finitestate = 1

        # Initialization
        if finitestate == 0:
            for i in range(0, N):
                betaHat[i, T-1] = 1/c[T-1]
        if finitestate == 1:
            for i in range(0, N):
                betaHat[i, T-1] = A[i, N]/(c[T]*c[T-1])

        # Backward Step
        if finitestate == 0:
            for j in range(T-1, 0, -1):
                for i in range(0, N):
                    betaHat[i, j-1] = (A[i, :]@(pX[:, j]*betaHat[:, j]))/c[j-1]
        if finitestate == 1:
            A = np.delete(A, N, 1)
            for j in range(T-1, 0, -1):
                for i in range(0, N):
                    betaHat[i, j-1] = (A[i, :]@(pX[:, j]*betaHat[:, j]))/c[j-1]

        return betaHat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
