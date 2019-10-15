import numpy as np
import scipy.stats as stats

from .baseline import *

def generate_pupil_data(event_onsets, fs=1000, baseline_lowpass=0.2, 
                        evoked_response_perc=0.02, response_fluct_sd=1,
                        prop_spurious_events=0.5, noise_amp=0.0005):
    """
    Generate artificial pupil data as a sum of slow baseline-fluctuations
    on which event-evoked responses are "riding". 
    
    Parameters:
    -----------
    
    event_onsets: list
        list of all events that evoke a response (in seconds)
        
    fs: float
        sampling rate in Hz
        
    baseline_lowpass: float
        cutoff for the lowpass-filter that defines the baseline
        (highest allowed frequency in the baseline fluctuations)
        
    evoked_response_perc: float
        amplitude of the pupil-response assproportion of the baseline 
    
    response_fluct_sd: float
        How much do the amplitudes of the individual events fluctuate?
        This is determined by drawing each individual pupil-response to 
        a single event from a (positive) normal distribution with mean as determined
        by `evoked_response_perc` and sd `response_fluct_sd` (in units of 
        `evoked_response_perc`).
        
    prop_spurious_events: float
        Add random events to the pupil signal. `prop_spurious_events` is expressed
        as proportion of the number of real events. 
        
    noise_amp: float
        Amplitude of random gaussian noise that sits on top of the simulated signal.
        Expressed in units of mean baseline pupil diameter.
    """
    T=np.array(event_onsets).max()+5 # stop 5 sec after last event
    n=int(np.ceil(T*fs)) # number of sampling points
    sy=np.zeros(n)       # pupil diameter 
    tx=np.linspace(0,T,n)     # time-vector in seconds
    nevents=len(event_onsets)
    
    # create baseline-signal 
    slack=int(0.50*n) # add slack to avoid edge effects of the filter
    x0=butter_lowpass_filter(np.random.rand(n+slack), baseline_lowpass, fs, 2)[slack:(n+slack)]
    x0=x0*1000+5000 # scale it up to a scale as usually obtained from eyetracker
    
    # create kernel and delta-functions for events
    kernel=pupil_kernel(duration=4,fs=fs)
    
    # real events regressor
    event_ix=(np.array(event_onsets)*fs).astype(np.int)
    deltas=np.zeros(n)
    #a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    delta_weights=stats.truncnorm.rvs(-1/response_fluct_sd,np.inf, loc=1, scale=response_fluct_sd, size=event_ix.size)
    deltas[event_ix]=delta_weights
    x1=np.convolve(deltas, kernel, mode="full")[0:n]

    # spurious events regressor
    sp_events=tx[ np.random.randint(low=0,high=n,size=int( nevents*prop_spurious_events )) ]
    sp_event_ix=(np.array(sp_events)*fs).astype(np.int)
    sp_deltas=np.zeros(n)
    sp_delta_weights=stats.truncnorm.rvs(-1/response_fluct_sd,np.inf, loc=1, scale=response_fluct_sd, size=sp_event_ix.size)
    sp_deltas[sp_event_ix]=sp_delta_weights
    x2=np.convolve(sp_deltas, kernel, mode="full")[0:n]
    
    amp=np.mean(x0)*evoked_response_perc # mean amplitude for the evoked response
    noise=noise_amp*np.mean(x0)*np.random.randn(n)
    
    sy = x0 + amp*x1 + amp*x2 + noise
    return (tx,sy,x0,deltas)


def get_dataset(ntrials=100, isi=2, rtdist=(1,0.5),fs=1000,pad=5.0):
    """
    Convenience function to run `generate_pupil_data()` with standard parameters.

    ntrials=100
    isi=2 # inter-stimulus-interval
    rtdist=(1,0.5) # reaction times (mean,sd)
    fs=1000 # sampling rate
    pad=5.0 ## padding for signal in seconds
    """
    stim_onsets=np.arange(ntrials)*isi+pad
    rts=stats.truncnorm.rvs( (0-rtdist[0])/rtdist[1], np.inf, loc=rtdist[0], scale=rtdist[1], size=ntrials)
    resp_onsets=stim_onsets+rts
    event_onsets=np.concatenate( (stim_onsets, resp_onsets) )

    tx,sy,baseline,response_coef=generate_pupil_data(event_onsets, fs=fs, 
                                                     baseline_lowpass=0.1, 
                                                     evoked_response_perc=0.001,
                                                     noise_amp=0.0002, 
                                                     prop_spurious_events=0.2)
    return tx,sy,baseline,event_onsets, response_coef
