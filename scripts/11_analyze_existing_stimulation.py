#!/usr/bin/env python
"""Characterize the provided synthetic stimulation recordings (no optimization claim)."""

import ast, csv, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.parameters import load_stimulation_parameters
from vbt.data.vep import VEPSubject
from vbt.evaluation.ezn import evaluate
from vbt.inference.reference_engine import read_brainvision
from vbt.stimulation.waveform import biphasic_waveform, charge_per_period

DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort")


def main():
    total_start=time.perf_counter(); out=ROOT/"outputs/stimulation_existing"; out.mkdir(parents=True,exist_ok=True)
    subjects=[]
    for i in range(1,31):
        subject=f"sub-{i:03d}"
        recordings=sorted((DATA/subject/"ses-02/ieeg").glob(f"{subject}_ses-02_task-simulatedstimulation_acq-VEPhypothesis_run-*_ieeg.vhdr"))
        if recordings: subjects.append((subject,recordings[0]))
    rows=[]; manifest=[]
    for subject,vhdr in subjects:
        started=time.perf_counter(); run=vhdr.name.split("_run-")[1].split("_")[0]
        base=DATA/"derivatives/tvb"/subject/"ses-02/VEPhypothesis"
        stim_path=base/"parameters"/f"{subject}_stimulation_parameters_run-{run}.tsv"
        source_path=base/f"{subject}_simulated_source_timeseries_run-{run}.npz"
        stim=load_stimulation_parameters(stim_path); loaded=time.perf_counter()
        archive=np.load(source_path); source=archive["source_signal"]; source_time=archive["time_steps"]
        pre=(source_time>=max(source_time[0],stim.onset-500))&(source_time<stim.onset)
        post=(source_time>=stim.onset)&(source_time<min(source_time[-1],stim.onset+500))
        baseline=np.median(np.abs(source[pre]),axis=0); response=np.max(np.abs(source[post]),axis=0)-baseline
        subject_data=VEPSubject.load(DATA,subject); labels=list(subject_data.region_names); ez=set(subject_data.model_parameters.ez_regions); pz=set(subject_data.model_parameters.pz_regions); truth=np.asarray([x in ez for x in labels])
        metric=evaluate(response,truth); features_at=time.perf_counter()
        raw,sfreq,channels=read_brainvision(vhdr); sample_onset=int(round(stim.onset*sfreq/1000.0)); window=int(round(0.5*sfreq)); sensor_pre=raw[max(0,sample_onset-window):sample_onset]; sensor_post=raw[sample_onset:min(raw.shape[0],sample_onset+window)]; sensor_delta=float(np.mean(np.sqrt(np.mean(sensor_post**2,axis=0))-np.sqrt(np.mean(sensor_pre**2,axis=0))))
        _,wave=biphasic_waveform(stim.stimulation_length,subject_data.simulator_parameters.dt,stim.onset,stim.period_samples,stim.amplitude,stim.pulse_width)
        denom=float(np.sum(np.abs(stim.weights))); ez_eng=float(np.sum(np.abs(stim.weights)[truth])/denom) if denom else 0.0; pz_mask=np.asarray([x in pz for x in labels]); pz_eng=float(np.sum(np.abs(stim.weights)[pz_mask])/denom) if denom else 0.0
        validation_at=time.perf_counter()
        row={"subject":subject,"run":run,"source_response_auroc":metric["auroc"],"source_response_ap":metric["average_precision"],"first_ez_rank":metric["first_ez_rank"],"oracle_k_recall":metric["oracle_k_recall"],"ez_target_engagement":ez_eng,"pz_target_engagement":pz_eng,"mean_sensor_rms_delta":sensor_delta,"amplitude_mA":stim.amplitude,"pulse_width_ms":stim.pulse_width,"charge_per_phase_uC":stim.amplitude*stim.pulse_width,"net_charge_one_period":charge_per_period(np.arange(wave.size)*subject_data.simulator_parameters.dt,wave,stim.period_samples,stim.onset),"data_load_s":loaded-started,"feature_s":features_at-loaded,"validation_s":validation_at-features_at,"total_s":validation_at-started}
        rows.append(row); manifest.append({"subject":subject,"primary_recording":str(vhdr),"source":str(source_path),"parameters":str(stim_path)})
    frame=pd.DataFrame(rows); frame.to_csv(out/"subject_metrics.csv",index=False); pd.DataFrame(manifest).to_csv(out/"stimulation_manifest.csv",index=False)
    summary={"subjects":len(rows),"recordings_policy":"lexicographically first valid VEP-hypothesis stimulation recording per subject","response_window":"500 model-time units before/after onset","endpoint":"peak absolute source response above median pre-stim baseline; AUROC vs synthetic EZ","metrics":{c:{"mean":float(frame[c].mean()),"median":float(frame[c].median()),"min":float(frame[c].min()),"max":float(frame[c].max())} for c in ["source_response_auroc","source_response_ap","first_ez_rank","oracle_k_recall","ez_target_engagement","mean_sensor_rms_delta"]},"timing_seconds":{"per_subject_mean":float(frame.total_s.mean()),"cohort_wall":time.perf_counter()-total_start},"claim":"provided synthetic stimulation response characterization; not optimization or therapeutic efficacy"}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
