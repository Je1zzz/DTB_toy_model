"""Biphasic pulse train used by the VEP stimulation generator."""
import numpy as np

def biphasic_waveform(simulation_length,dt,onset,period_samples,amplitude,pulse_width):
    time=np.arange(0,float(simulation_length),float(dt)); wave=np.zeros(time.size)
    # TVB PulseTrain uses the saved values in model-time/sample units.
    for start in np.arange(float(onset),float(simulation_length),float(period_samples)):
        wave[(time>=start)&(time<start+pulse_width)]+=amplitude
        wave[(time>=start+pulse_width)&(time<start+2*pulse_width)]-=amplitude
    return time,wave

def charge_per_period(time,wave,period_samples,onset):
    mask=(time>=onset)&(time<onset+period_samples)
    if mask.sum()<2: return 0.
    return float(np.sum(wave[mask]) * np.median(np.diff(time[mask])))
