import pickle
import glob
import os
import sys
import time
import numpy as np
import pandas as pd 
import multiprocessing as mp 
import openmdao.api as om
from weis.aeroelasticse import FileTools
# import raft
import matplotlib.pyplot as plt
import matplotlib

#----------------------------------------------------------------------------------------------
#---------------------------------------- Log ---------------------------------------------------
#----------------------------------------------------------------------------------------------
# This function loads the openmdao sql file and does most of the work here
def load_OMsql(log):
    print('loading {}'.format(log))
    cr = om.CaseReader(log)
    rec_data = {}
    driver_cases = cr.list_cases('driver')
    cases = cr.get_cases('driver')
    for case in cases:
        for key in case.outputs.keys():
            if key not in rec_data:
                rec_data[key] = []
            rec_data[key].append(case[key])
        
    return rec_data

rec_data = load_OMsql('outputs/02_control_opt/log_opt.sql')

# Search for keys in rec_data, it has every openmdao input/output from every iteration, so it's large
for key in rec_data:
    if 'float' in key:
        print(key)

# Usually, you need to squeeze the rec data
np.squeeze(rec_data['tune_rosco_ivc.Kp_float'])

#----------------------------------------------------------------------------------------------
#---------------------------------------- out -------------------------------------------------
#----------------------------------------------------------------------------------------------
import os, subprocess
from ROSCO_toolbox.ofTools.fast_io import output_processing
from ROSCO_toolbox.ofTools.util import spectral
import pandas as pd
import matplotlib.pyplot as plt

i_fig = 0
outfiles = [
    'outputs/02_control_opt/IEA15_0.out',
    'outputs/02_control_opt/IEA15_1.out',
    'outputs/02_control_opt/IEA15_2.out',
]
output_ext = '.out'
plt.rcParams["figure.figsize"] = [9,7]
ROSCO = True   # Include RO.dbg file in output dict
ROSCO2 = True  # Include RO.dbg2 file in output dict
#  Define Plot cases 
cases = {}
cases['Gen. Speed Sigs.'] = ['Wind1VelX', 'GenTq', 'GenSpeed','GenPwr','NacYaw']#,'PtfmPitch','PtfmYaw','NacYaw']
cases['Plt. Control Sigs.'] = ['RtVAvgxh', 'BldPitch1', 'Fl_Pitcom', 'PC_MinPit','WE_Vw']
cases['Platform Motion'] = ['PtfmSurge', 'PtfmSway', 'PtfmHeave', 'PtfmPitch','PtfmRoll','PtfmYaw']
# cases['Rot Thrust'] = ['RtVAvgxh','BldPitch1','RotThrust']

op = output_processing.output_processing()
op_RO = output_processing.output_processing()
op_RO2 = output_processing.output_processing()


fast_out = []
fast_out = op.load_fast_out(outfiles, tmin=0)
if ROSCO:
    # Rosco outfiles
    r_outfiles = [out.split('.out')[0] + '.RO.dbg' for out in outfiles]
    rosco_out = op_RO.load_fast_out(r_outfiles, tmin=0)
    
    if ROSCO2:
        r_outfiles = [out.split('.out')[0] + '.RO.dbg2' for out in outfiles]
        rosco_out2 = op_RO2.load_fast_out(r_outfiles, tmin=0)
  
# Combine outputs
if ROSCO:
    comb_out = [None] * len(fast_out)
    for i, (r_out, f_out) in enumerate(zip(rosco_out,fast_out)):
        r_out.update(f_out)
        comb_out[i] = r_out
    if ROSCO2:
        for i, (r_out2, f_out) in enumerate(zip(rosco_out2,comb_out)):
            r_out2.update(f_out)
            comb_out[i] = r_out2
else:
    comb_out = fast_out

# Plot
fig, ax = op.plot_fast_out(comb_out,cases, showplot=True)
if True:  # Print!
    save_fig_dir = 'outputs/02_control_opt'
    for f in fig:
        f.savefig(os.path.join(save_fig_dir,'ts'+str(i_fig)))
        i_fig += 1

df = pd.DataFrame()
for i in range(len(outfiles)):
    fq, y, _ = spectral.fft_wrap(
                        fast_out[i]['Time'], fast_out[i]['RotThrust'], averaging='Welch', averaging_window='Hamming', output_type='psd')
    plt.plot(fq,np.sqrt(y))
    df['fq_'+str(i)] = fq
    df['psd_'+str(i)] = y    
#     fq, y, _ = spectral.fft_wrap(
#                         fast_out[i]['Time'], fast_out[i]['RtVAvgxh'], averaging='Welch', averaging_window='Hamming', output_type='psd')
#     plt.plot(fq,np.sqrt(y))
#     fq, y, _ = spectral.fft_wrap(
#                     fast_out[i]['Time'], fast_out[i]['Wind1VelX'], averaging='Welch', averaging_window='Hamming', output_type='psd')
#     plt.plot(fq,np.sqrt(y))   
plt.yscale('log')
plt.xscale('log')
plt.xlim([1e-2,10])
plt.grid('True')
plt.xlabel('Freq. (Hz)')
plt.ylabel('PSD')

#----------------------------------------------------------------------------------------------
#---------------------------------------- Case Matrix -----------------------------------------
#----------------------------------------------------------------------------------------------
import ruamel_yaml as ry
def plot_tss(dfs,channels):

    fig, axs = plt.subplots(len(channels),1)
    fig.set_size_inches(12,2*len(channels))

    if len(channels) == 1:
        axs = [axs]

    axs = axs.flatten()
    
    for df in dfs:

        for i_chan, chan in enumerate(channels):
            axs[i_chan].plot(df.Time,df[chan])
            axs[i_chan].set_ylabel(chan)
            axs[i_chan].grid()

        axs[-1].set_xlabel('Time')
        
    [a.set_xticklabels('') for a in axs[:-1]]
    [a.grid(True) for a in axs]
        
    fig.patch.set_facecolor('white')
    fig.align_ylabels()

    
    return fig, axs

# Function for reading case matrix
def read_cm(fname_case_matrix):
    cm_dict = FileTools.load_yaml(fname_case_matrix, package=1)
    cnames = []
    for c in list(cm_dict.keys()):
        if isinstance(c,ry.comments.CommentedKeySeq):
            cnames.append(tuple(c))
        else:
            cnames.append(c)

    cm = pd.DataFrame(cm_dict, columns=cnames)

    cm[('DLC','Label')].unique()

    dlc_inds = {}

    for dlc in cm[('DLC','Label')].unique():
        dlc_inds[dlc] = cm[('DLC','Label')] == dlc
        
    return cm, dlc_inds

fname_case_matrix = 'outputs/02_control_opt/case_matrix.yaml'
cm, dlc_inds = read_cm(fname_case_matrix)

cm[('DLC','Label')].unique()
dlc_inds = {}
for dlc in cm[('DLC','Label')].unique():
    dlc_inds[dlc] = cm[('DLC','Label')] == dlc

ss = pd.read_pickle("outputs/02_control_opt/iea15mw.pkl")