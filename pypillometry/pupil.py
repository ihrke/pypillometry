import numpy as np

def pupil_kernel(duration=4, fs=1000, npar=10.1, tmax=930.0):
    n=int(duration*fs)
    t = np.linspace(0,duration, n, dtype = np.float)*1000 # in ms
    h = t**(npar) * np.exp(-npar*t / tmax)   #Erlang gamma function Hoek & Levelt (1993)
    return h/h.max() # rescale to height=1
