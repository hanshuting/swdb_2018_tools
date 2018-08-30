import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

def get_trial_len(dataset):
    '''this function gets the starts, ends and the length of each image repetition sequence from the given dataset'''
    
    stim_table = dataset.get_stimulus_table()
    im_name = stim_table['image_name'].unique()
    
    # get trial length for each image
    trial_start = []
    trial_end = []
    trial_len = []
    # start_time = []
    # end_time = []

    for i, im in enumerate(im_name):
        t = stim_table['image_name']==im
        trial_diff = t[:-1].values*1 - t[1:].values*1
        if t[0]==1:
            trial_diff = np.insert(trial_diff,0,-1)
        if t[len(t)-1]==1:
            trial_diff = np.insert(trial_diff,len(trial_diff),1)
        trial_start.append(np.where(trial_diff==-1)[0])
        trial_end.append(np.where(trial_diff==1)[0])
        trial_len.append(trial_end[i]-trial_start[i]+1)
        # start_time.append(stim_table['start_time'].values[np.where(trial_diff==1)])
        # end_time.append(stim_table['end_time'].values[np.where(trial_diff==1)])
    
        result_dict = {'image':[im_name[0]]*len(trial_start[0]), 'trial_start':list(trial_start[0]), 'trial_end':list(trial_end[0]), 'trial_length':list(trial_len[0])} #, 'start_time':list(start_time[0]), 'end_time':list(end_time[0])}
    for i in range(len(im_name)-1):
        result_dict['image'] += [im_name[i+1]]*len(trial_start[i+1])
        result_dict['trial_start'] += list(trial_start[i+1])
        result_dict['trial_end'] += list(trial_end[i+1])
        result_dict['trial_length'] += list(trial_len[i+1])
        # result_dict['start_time'] += list(start_time[i+1])
        # result_dict['end_time'] += list(end_time[i+1])
    
    result = pd.DataFrame(result_dict)
    result = result[['image','trial_start','trial_end','trial_length']]
    
    return result


def plot_cell_tuning_curve(data, num_cell_per_row = 5, cmap = cm.cool):
    '''
    plot individual cell tuning curves over all repeats in trials
    INPUT:
        - data: num_cell x num_stim x num_repeat matrix with average response amplitude
        - num_cell_per_row: (optional) number of suplots per row, default = 5
        - cmap: (optional) colormap, default = cm.cool
    '''
    
    # get data dimension
    dims = data.shape
    num_cell = dims[0]
    len_cutoff = dims[2]
    
    # set colors
    color_idx = np.linspace(0, 1, len_cutoff)
    
    # loop over cells
    num_row = int(np.ceil(num_cell/num_cell_per_row))

    # create plot
    fig, axes = plt.subplots(num_row, num_cell_per_row)
    fig.set_size_inches(12, 12)

    for row in range(num_row):

        cell_start = row*num_cell_per_row
        cell_end = np.max([row*num_cell_per_row+1, num_cell])
        for cell, ax in zip(np.arange(cell_start, cell_end), axes[row,:]): # loop over cells

            for i, c in zip(range(len_cutoff),color_idx): # loop over repeats
                ax.plot(data[cell,:,i], color=cmap(c))
                ax.set_ylabel('cell' + str(cell))
    
    fig.text(0.5, 0, 'image index', ha='center', fontsize=16)
    fig.text(0.04, 0.5, 'average response', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


    
def plot_repeat_response(dff, trial_len, stim_table, ophys_ts, len_cutoff, repeat_start, cell, repeat, repeat_ts):
    
    '''
    # TODO: CLEAN UP THIS FUNCTION
    plot individual repeat response over all trials and all stimulus for the given cell and repeat number
    INPUT:
        - dff: num_cell x num_frame dF/F traces matrix
        - trial_len: trial start, end, and length dataframe returned by function get_trial_len
        - stim_table: stimulus table dataframe returned by dataset.get_stimulus_table()
        - ophys_ts: imaging frame timestamps
        - len_cutoff: minimum repeats each trial needs to contain
        - cell: cell index to plot, start from 0
        - repeat_start: from which repeat in the trial to start from, it's a number that's smaller than len_cutoff
        - repeat: which repeat to plot, start from 0
        - repeat_ts: timestamps in seconds for repeats
    '''
    
    stim_duration_frame = int(len(repeat_ts)/2)

    fig, axes = plt.subplots(2, 4, figsize=[12,6])

    # plot repeat responses for each image template
    im_name = trial_len['image'].unique()
    for im, ax in zip(im_name, axes.reshape(-1)):

        im_trials = np.where((trial_len['image']==im) & (trial_len['trial_length']>=len_cutoff))[0]
        trial_rsp_mat = np.zeros([len(im_trials), len_cutoff, int(stim_duration_frame)*2])

        for i, trial_block_id in enumerate(im_trials): # loop over trials for the image template

            for j, repeat_ind in enumerate(np.arange(trial_len['trial_start'][trial_block_id]+repeat_start, trial_len['trial_start'][trial_block_id]+len_cutoff+repeat_start, 1)): # loop over flashes in each trial

                # define frames for current trial block, current repeat
                flash_start = stim_table['start_time'][repeat_ind]
                stim_start = np.argmax(ophys_ts >= flash_start)
                frames = np.arange(stim_start-stim_duration_frame, stim_start+stim_duration_frame)

                # store result
                trial_rsp_mat[i, j, :] = dff[cell, frames]

        # plot individual repeats
        for i in range(trial_rsp_mat.shape[0]):
            ax.plot(repeat_ts[:stim_duration_frame*2], trial_rsp_mat[i, repeat, :], color=(0.7, 0.7, 0.7))

        # plot average response
        ax.plot(repeat_ts[:stim_duration_frame*2], trial_rsp_mat[:, repeat, :].mean(axis=0), color='r', linewidth=1.5)

        # plot stim time
        ax.axvline(x=repeat_ts[int(stim_duration_frame)], color='k', linewidth=1, linestyle='--')
        ax.set_xlim([repeat_ts[0], repeat_ts[-1]])
        ax.set_ylabel(im)

    plt.suptitle('cell ' + str(cell) + ', repeat ' + str(repeat))
    plt.show()