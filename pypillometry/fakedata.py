"""
fakedata.py
====================================
Generate artificial pupil-data.
"""
import numpy as np
import scipy.stats as stats

from .baseline import *
from .pupil import *

def generate_pupil_data(event_onsets, fs=1000, pad=5000, baseline_lowpass=0.2, 
                        evoked_response_perc=0.02, response_fluct_sd=1,
                        prf_npar=(10.35,0), prf_tmax=(917.0,0),
                        prop_spurious_events=0.2, noise_amp=0.0005):
    """
    Generate artificial pupil data as a sum of slow baseline-fluctuations
    on which event-evoked responses are "riding". 
    
    Parameters
    -----------
    
    event_onsets: list
        list of all events that evoke a response (in seconds)
        
    fs: float
        sampling rate in Hz
    pad: float
        append `pad` milliseconds of signal after the last event is decayed    
    baseline_lowpass: float
        cutoff for the lowpass-filter that defines the baseline
        (highest allowed frequency in the baseline fluctuations)
        
    evoked_response_perc: float
        amplitude of the pupil-response as proportion of the baseline 
    
    response_fluct_sd: float
        How much do the amplitudes of the individual events fluctuate?
        This is determined by drawing each individual pupil-response to 
        a single event from a (positive) normal distribution with mean as determined
        by `evoked_response_perc` and sd `response_fluct_sd` (in units of 
        `evoked_response_perc`).
    prf_npar: tuple (float,float)
        (mean,std) of the npar parameter from :py:func:`pypillometry.pupil.pupil_kernel()`. 
        If the std is exactly zero, then the mean is used for all pupil-responses.
        If the std is positive, npar is taken i.i.d. from ~ normal(mean,std) for each event.
    prf_tmax: tuple (float,float)
        (mean,std) of the tmax parameter from :py:func:`pypillometry.pupil.pupil_kernel()`. 
        If the std is exactly zero, then the mean is used for all pupil-responses.
        If the std is positive, tmax is taken i.i.d. from ~ normal(mean,std) for each event.
    prop_spurious_events: float
        Add random events to the pupil signal. `prop_spurious_events` is expressed
        as proportion of the number of real events. 
        
    noise_amp: float
        Amplitude of random gaussian noise that sits on top of the simulated signal.
        Expressed in units of mean baseline pupil diameter.
        
    
    Returns
    --------
    
    tx, sy: np.array
        time and simulated pupil-dilation (n)
    x0: np.array
        baseline (n)
    delta_weights: np.array
        pupil-response strengths (len(event_onsets))
    """
    nevents=len(event_onsets)
    ## npar
    if prf_npar[1]==0: # deterministic parameter
        npars=np.ones(nevents)*prf_npar[0]
    else:
        npars=np.random.randn(nevents)*prf_npar[1]+prf_npar[0]

    ## tmax
    if prf_tmax[1]==0: # deterministic parameter
        tmaxs=np.ones(nevents)*prf_tmax[0]
    else:
        tmaxs=np.random.randn(nevents)*prf_tmax[1]+prf_tmax[0]

    if np.any(npars<=0):
        raise ValueError("npar must be >0")
    if np.any(tmaxs<=0):
        raise ValueError("tmax must be >0")

    # get maximum duration of one of the PRFs
    maxdur=pupil_get_max_duration(npars.min(), tmaxs.max())

    T=np.array(event_onsets).max()+maxdur+pad # stop pad millisec after last event
    n=int(np.ceil(T/1000.*fs)) # number of sampling points
    sy=np.zeros(n)       # pupil diameter 
    tx=np.linspace(0,T,n)     # time-vector in milliseconds

    # create baseline-signal 
    slack=int(0.50*n) # add slack to avoid edge effects of the filter
    x0=butter_lowpass_filter(np.random.rand(n+slack), baseline_lowpass, fs, 2)[slack:(n+slack)]
    x0=x0*1000+5000 # scale it up to a scale as usually obtained from eyetracker


    ### real events regressor
    ## scaling
    event_ix=(np.array(event_onsets)/1000.*fs).astype(np.int)
    #a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    delta_weights=stats.truncnorm.rvs(-1/response_fluct_sd,np.inf, loc=1, scale=response_fluct_sd, size=event_ix.size)
    x1=np.zeros_like(sy)

    for i,ev in enumerate(event_onsets):
        # create kernel and delta-functions for events
        kernel=pupil_kernel(duration=maxdur,fs=fs,npar=npars[i], tmax=tmaxs[i])
        x1[event_ix[i]:(event_ix[i]+kernel.size)]=x1[event_ix[i]:(event_ix[i]+kernel.size)]+kernel*delta_weights[i]

    ## spurious events regressor

    sp_event_ix=np.random.randint(low=0,high=np.ceil((T-maxdur-pad)/1000.*fs),size=int( nevents*prop_spurious_events ))
    sp_events=tx[ sp_event_ix ]
    n_sp_events=sp_events.size

    ## npar
    if prf_npar[1]==0: # deterministic parameter
        npars=np.ones(n_sp_events)*prf_npar[0]
    else:
        npars=np.random.randn(n_sp_events)*prf_npar[1]+prf_npar[0]

    ## tmax
    if prf_tmax[1]==0: # deterministic parameter
        tmaxs=np.ones(n_sp_events)*prf_tmax[0]
    else:
        tmaxs=np.random.randn(n_sp_events)*prf_tmax[1]+prf_tmax[0]


    ## scaling
    sp_delta_weights=stats.truncnorm.rvs(-1/response_fluct_sd,np.inf, loc=1, scale=response_fluct_sd, size=sp_event_ix.size)
    x2=np.zeros_like(sy)

    for i,ev in enumerate(sp_events):
        # create kernel and delta-functions for events
        kernel=pupil_kernel(duration=maxdur,fs=fs,npar=npars[i], tmax=tmaxs[i])
        x2[sp_event_ix[i]:(sp_event_ix[i]+kernel.size)]=x2[sp_event_ix[i]:(sp_event_ix[i]+kernel.size)]+kernel*sp_delta_weights[i]

    amp=np.mean(x0)*evoked_response_perc # mean amplitude for the evoked response
    noise=noise_amp*np.mean(x0)*np.random.randn(n)

    sy = x0 + amp*x1 + amp*x2 + noise

    return (tx,sy,x0,delta_weights)


def get_dataset(ntrials=100, isi=2000, rtdist=(1000,500),fs=1000,pad=5000, **kwargs):
    """
    Convenience function to run :py:func:`generate_pupil_data()` with standard parameters.
    Parameters
    -----------
    
    ntrials:int
        number of trials
    isi: float
        inter-stimulus interval in milliseconds
    rtdist: tuple (float,float)
        mean and std of a (truncated at zero) normal distribution to generate response times
    fs: float
        sampling rate
    pad: float
        padding before the first and after the last event in seconds
    kwargs: dict
        arguments for :py:func:`pypillometry.fakedata.generate_pupil_data()`

    
    Returns
    --------
        
    tx, sy: np.array
        time and simulated pupil-dilation (n)
    baseline: np.array
        baseline (n)
    event_onsets: np.array
        timing of the simulated event-onsets (stimuli and responses not separated)
    response_coef: np.array
        pupil-response strengths (len(event_onsets))
    """
    stim_onsets=np.arange(ntrials)*isi+pad
    rts=stats.truncnorm.rvs( (0-rtdist[0])/rtdist[1], np.inf, loc=rtdist[0], scale=rtdist[1], size=ntrials)
    resp_onsets=stim_onsets+rts
    event_onsets=np.concatenate( (stim_onsets, resp_onsets) )

    kwargs.update({"fs":fs})
    tx,sy,baseline,response_coef=generate_pupil_data(event_onsets, **kwargs)
    return tx,sy,baseline,event_onsets, response_coef
