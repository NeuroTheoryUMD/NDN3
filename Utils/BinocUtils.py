"""Neural deep network situtation-specific utils by Dan"""

#from __future__ import division
import numpy as np
import scipy.io as sio
import NDN3.NDN as NDN
import NDN3.NDNutils as NDNutils
import NDN3.Utils.DanUtils as DU
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.preprocessing import normalize as sk_normalize


################
def plot_tfilters( ndnmod, kts = None, to_plot=True  ):
    """Can pass in weights to relevant layer in first argument, as well as NDN model.
    Will use default tkerns variable, but this can also be passed in as kts argument."""

    ntk = kts.shape[1]
    if type(ndnmod) is np.ndarray:  # then passing in filters and need more information
        ws = deepcopy(ndnmod)     
    else:        
        ws = deepcopy(ndnmod.networks[0].layers[0].weights)
    
    if len(ws.shape) > 2:
        nx, ntk2, numcells = ws.shape
        ws2 = ws
    else:
        nx = ws.shape[0] // ntk
        numcells = ws.shape[1]
        ws2 = np.reshape(ws, [nx, ntk, numcells])
                         
    ks = np.expand_dims(kts,axis=0)@ws2
    if to_plot:
        DU.plot_filters(filters=ks)
    else:
        return ks

 
def compute_binocular_filters(binoc_mod, to_plot=True):

    # Find binocular layer
    blayer, bnet = None, None
    for mm in range(len(binoc_mod.networks)):
        for nn in range(len(binoc_mod.networks[mm].layers)):
            if binoc_mod.network_list[mm]['layer_types'][nn] == 'biconv':
                if nn < len(binoc_mod.networks[mm].layers) - 1:
                    bnet, blayer = mm, nn + 1
                elif mm < len(binoc_mod.networks) - 1:
                    bnet, blayer = mm + 1, 0  # split in hierarchical network
    assert blayer is not None, 'biconv layer not found'

    NF = binoc_mod.networks[0].layers[blayer].output_dims[0]
    Nin = binoc_mod.networks[0].layers[blayer].input_dims[0]
    NX = binoc_mod.networks[0].layers[blayer].filter_dims[1]
    ks1 = DU.compute_spatiotemporal_filters(binoc_mod)
    ws = np.reshape(binoc_mod.networks[0].layers[blayer].weights, [NX, Nin, NF])
    num_lags = binoc_mod.networks[0].layers[0].input_dims[0]
    if binoc_mod.networks[0].layers[0].filter_dims[1] > 1:  # then not temporal layer
        filter_dims = [num_lags, binoc_mod.networks[0].layers[0].filter_dims[1]]
    else:
        filter_dims = [num_lags, binoc_mod.networks[0].layers[1].filter_dims[1]]
    nfd = [filter_dims[0], filter_dims[1] + NX]
    # print(filter_dims, nfd)
    Bfilts = np.zeros(nfd + [NF, 2])
    for nn in range(NX):
        Bfilts[:, np.add(range(filter_dims[1]), nn), :, 0] += np.reshape(
            np.matmul(ks1, ws[nn, range(Nin // 2), :]), [filter_dims[1], filter_dims[0], NF])
        Bfilts[:, np.add(range(filter_dims[1]), nn), :, 1] += np.reshape(
            np.matmul(ks1, ws[nn, range(Nin // 2, Nin), :]), [filter_dims[1], filter_dims[0], NF])
    bifilts = np.concatenate((Bfilts[:, :, :, 0], Bfilts[:, :, :, 1]), axis=1)
    if to_plot:
        DU.plot_filters(filters=bifilts, flipxy=True)
    return bifilts


########### FILE MANAGEMENT FOR BINOCULAR MODELS / DATA ###########
def binocular_matlab_export(binoc_mod, filename):
    """Export binocular model (including filter calc) to .mat file"""

    import scipy.io as sio
    matdata = {}
    for nn in range(binoc_mod.num_networks):
        for ll in range(len(binoc_mod.networks[nn].layers)):
            wstring = 'ws' + str(nn) + str(ll)
            matdata[wstring] = binoc_mod.networks[nn].layers[ll].weights
            bstring = 'bs' + str(nn) + str(ll)
            matdata[bstring] = binoc_mod.networks[nn].layers[ll].biases

    bfilts = compute_binocular_filters(binoc_mod=binoc_mod, to_plot=False)
    matdata['bfilts'] = bfilts
    sio.savemat(filename, matdata)


def binocular_data_import( datadir, expt_num ):
    """Usage: stim, Robs, DFs, used_inds = binocular_data_import( datadir, expt_num )

    Inputs:
        datadir: directory on local drive where datafiles are
        expt_num: the experiment number (1-12) representing the binocular experiments we currently have. All
                    datafiles are called 'BS2expt?.mat'. Note that numbered from 1 (not zero)
    
    Outputs:
        stim: formatted as NT x 72 (stimuli in each eye are cropped to NX=36). It will be time-shifted by 1 to 
                eliminate 0-latency stim
        Robs: response concatenating all SUs and MUs, NT x NC. NSU is saved as part of Eadd_info
        DFs:  data_filters for experiment, also NT x NC (note MU datafilters are initialized to 1)
        used_inds: indices overwhich data is valid according to initial data parsing (adjusted to python) 
        Eadd_info: dictionary containing all other relevant info for experiment
    """

    # Constants that are pretty much set for our datasets
    stim_trim = np.concatenate( (range(3,39), range(45,81)))
    time_shift = 1
    NX = 36

    # Read all data into memory
    filename = 'B2Sexpt'+ str(expt_num) + '.mat'
    Bmatdat = sio.loadmat(filename)
    stim = NDNutils.shift_mat_zpad(Bmatdat['stim'][:,stim_trim], time_shift, 0)

    #NX = int(stim.shape[1]) // 2
    RobsSU = Bmatdat['RobsSU']
    RobsMU = Bmatdat['RobsMU']
    #MUA = Bmatdata[nn]['RobsMU']
    #SUA = Bmatdata[nn]['RobsSU']
    NTtot, numSUs = RobsSU.shape

    data_filtersSU = Bmatdat['SUdata_filter']
    data_filtersMU = np.ones( RobsMU.shape, dtype='float32')  # note currently no MUdata filter but their needs to be....

    Robs = np.concatenate( (RobsSU, RobsMU), axis=1 )
    DFs = np.concatenate( (data_filtersSU, data_filtersMU), axis=1 )

    # Valid and cross-validation indices
    used_inds = np.add(np.transpose(Bmatdat['used_inds'])[0,:], -1) # note adjustment for python v matlab indexing
    Ui_analog = Bmatdat['Ui_analog'][:,0]  # these are automaticall in register
    XiA_analog = Bmatdat['XiA_analog'][:,0]
    XiB_analog = Bmatdat['XiB_analog'][:,0]
    # two cross-validation datasets -- for now combine
    Xi_analog = XiA_analog+XiB_analog  # since they are non-overlapping, will make 1 in both places

    # Derive full-dataset Ui and Xi from analog values
    Ui = np.intersect1d(used_inds, np.where(Ui_analog > 0)[0])
    Xi = np.intersect1d(used_inds, np.where(Xi_analog > 0)[0])

    NC = Robs.shape[1]
    NT = len(used_inds)

    dispt_raw = Bmatdat['all_disps'][:,0]
    # this has the actual disparity values, which are at the resolution of single bars, and centered around the neurons
    # disparity (sometime shifted to drive neurons well)
    # Sometimes a slightly disparity is used, so it helps to round the values at some resolution
    dispt = np.round(dispt_raw*100)/100
    disp_list = np.unique(dispt)
    # where it is -1009 this corresponds to a blank frame
    # where it is -1005 this corresponds to uncorrelated images between the eyes
    corrt = Bmatdat['all_corrs'][:,0]
    frs = Bmatdat['all_frs'][:,0]
    rep_inds = np.add(Bmatdat['rep_inds'], -1)  

    print( "Expt %d: %d SUs, %d total units, %d out of %d time points used."%(expt_num, numSUs, NC, NT, NTtot))
    #print(len(disp_list), 'different disparities:', disp_list)

    Eadd_info = {
        'Ui_analog': Ui_analog, 'XiA_analog': XiA_analog, 'XiB_analog': XiB_analog, 'Xi_analog': Xi_analog,
        'Ui': Ui, 'Xi': Xi, # these are for use with data_filters and inclusion of whole experiment,
        'NSU': numSUs, 'NC': NC,
        'dispt': dispt, 'corrt': corrt, 'disp_list': disp_list, 
        'frs': frs, 'rep_inds': rep_inds}

    return stim, Robs, DFs, used_inds, Eadd_info


def binocular_data_import_cell( datadir, expt_num, cell_num ):
    """Usage: stim, Robs, used_inds_cell, UiC, XiC = binocular_data_import_cell( datadir, expt_num, cell_num)

    Imports data for one cell: otherwise identitical to binocular_data_import. Takes data_filter for the 
    cell into account so will not need datafilter. Also, all indices will be adjusted accordingly.

    Inputs:
        datadir: directory on local drive where datafiles are
        expt_num: the experiment number (1-12) representing the binocular experiments we currently have. All
                    datafiles are called 'BS2expt?.mat'. Note that numbered from 1 (not zero)
        cell_num: cell number to analyze
    
    Outputs:
        stim: formatted as NT x 72 (stimuli in each eye are cropped to NX=36). It will be time-shifted by 1 to 
                eliminate 0-latency stim
        Robs: response concatenating all SUs and MUs, NT x NC. NSU is saved as part of Eadd_info
        used_inds_cell: indices overwhich data is valid for that particular cell
        UiC, XiC: cross-validation indices for that cell, based on used_inds_cell
        Eadd_info: dictionary containing all other relevant info for experiment
    """

    stim, Robs_all, DFs, used_inds, Einfo = binocular_data_import( datadir, expt_num )

    cellspecificdata = np.where(DFs[:, cell_num-1] > 0)[0]
    used_cell = np.intersect1d(used_inds, cellspecificdata)
    NT = len(used_cell)
    Ui = np.where(Einfo['Ui_analog'][used_cell] > 0)[0]

    Xi_analog = Einfo['XiA_analog']+Einfo['XiB_analog']
    XVi1 = np.where(Einfo['XiA_analog'][used_cell] > 0)[0]
    XVi2 = np.where(Einfo['XiB_analog'][used_cell] > 0)[0]
    Xi = np.where(Xi_analog[used_cell] > 0)[0]

    Robs = np.expand_dims(Robs_all[used_cell, cell_num-1], 1)

    # Add some things to Einfo
    Einfo['used_inds_all'] = used_inds
    Einfo['XiA_cell'] = XVi1
    Einfo['XiB_cell'] = XVi2

    print( "Adjusted for cell %d: %d time points"%(cell_num, NT))

    return stim, Robs, used_cell, Ui, Xi, Einfo

