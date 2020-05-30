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

    assert kts is not None, 'Must include tkerns.'
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
    else:
        return bifilts


def compute_binocular_tfilters(binoc_mod, kts=None, to_plot=True):

    assert kts is not None, 'Must include tkerns.'
    BFs = compute_binocular_filters( binoc_mod, to_plot=False)
    Bks = np.transpose(np.tensordot(kts, BFs, axes=[1, 0]), (1,0,2))

    if to_plot:
        DU.plot_filters(filters=Bks)
    else:
        return Bks


###################### DISPARITY PROCESSING ######################
def disparity_matrix( dispt, corrt ):

    # last two colums will be uncorrelated (-1005) and blank (-1009)

    dlist = np.unique(dispt)[2:]  # this will exclude -1009 (blank) and -1005 (uncor)
    ND = len(dlist)

    dmat = np.zeros([dispt.shape[0], 2*ND+2])
    dmat[np.where(dispt == -1009)[0], -1] = 1
    dmat[np.where(dispt == -1005)[0], -2] = 1
    for dd in range(len(dlist)):
        dmat[np.where((dispt == dlist[dd]) & (corrt > 0))[0], dd] = 1
        dmat[np.where((dispt == dlist[dd]) & (corrt < 0))[0], ND+dd] = 1

    return dmat


def disparity_tuning( Einfo, r, used_inds=None, num_dlags=8, fr1or3=3, to_plot=False ):

    if used_inds is None:
        used_inds = range(len(r))

    dmat = disparity_matrix( Einfo['dispt'], Einfo['corrt'])
    ND = (dmat.shape[1]-2) // 2
    
    # Weight all by their frequency of occurance
    
    if (fr1or3 == 3) or (fr1or3 == 1):
        frs_valid = Einfo['frs'] == fr1or3
    else:
        frs_valid = Einfo['frs'] > 0
    
    to_use = frs_valid[used_inds]
    dmatN = dmat / np.mean(dmat[used_inds[to_use],:], axis=0) * np.mean(dmat[used_inds[to_use],:])
    Xmat = NDNutils.create_time_embedding( dmatN[:, range(ND*2)], [num_dlags, 2*ND, 1])[used_inds, :]
    # uncorrelated response
    Umat = NDNutils.create_time_embedding( dmatN[:, [-2]], [num_dlags, 1, 1])[used_inds, :]

    if len(r) > len(used_inds):
        resp = r[used_inds] 
    else: 
        resp = r
                          
    Nspks = np.sum(resp[to_use, :], axis=0)
    Dsta = np.reshape( Xmat[to_use, :].T@resp[to_use], [2*ND, num_dlags] ) / Nspks
    Usta = (Umat[to_use, :].T@resp[to_use])[:,0] / Nspks
    
    # Rudimentary analysis
    best_lag = np.argmax(np.max(Dsta[range(ND),:], axis=0))
    Dtun = np.reshape(Dsta[:, best_lag], [2, ND]).T
    uncor_resp = Usta[best_lag]
    
    Dinfo = {'Dsta':Dsta, 'Dtun': Dtun, 'uncor_resp': uncor_resp, 'best_lag': best_lag, 'uncor_sta': Usta}
                    
    if to_plot:
        DU.subplot_setup(1,2)
        plt.subplot(1,2,1)
        DU.plot_norm(Dsta.T-uncor_resp, cmap='bwr')
        plt.plot([ND-0.5,ND-0.5], [-0.5, num_dlags-0.5], 'k')
        plt.plot([-0.5, 2*ND-0.5], [best_lag, best_lag], 'k--')
        plt.subplot(1,2,2)
        plt.plot(Dtun)
        plt.plot(-Dtun[:,1]+2*uncor_resp,'m--')

        plt.plot([0, ND-1], [uncor_resp, uncor_resp], 'k')
        plt.xlim([0, ND-1])
        plt.show()
        
    return Dinfo


def disparity_predictions( Einfo, resp, indxs=None, num_dlags=8, fr1or3=None, spiking=True, opt_params=None ):
    """Calculates a prediction of the disparity (and timing) signals that can be inferred from the response
    by the disparity input alone. This puts a lower bound on how much disparity is driving the response, although
    practically speaking will generate the same disparity tuning curves.
    
    Usage: Dpred, Tpred = disparity_predictions( Einfo, resp, indxs, num_dlags=8, spiking=True, opt_params=None )

    Inputs: Indices gives data range to fit to.
    Outputs: Dpred and Tpred will be length of entire experiment -- not just indxs
    """

    # Process disparity into disparty and timing design matrix
    dmat = disparity_matrix( Einfo['dispt'], Einfo['corrt'] )
    ND2 = dmat.shape[1]
    if indxs is None:
        indxs = range(dmat.shape[0])

    # everything but blank
    Xd = NDNutils.create_time_embedding( dmat[:, :-1], [num_dlags, ND2-1, 1])[indxs,:]
    # blank
    Xb = NDNutils.create_time_embedding( dmat[:, -1], [num_dlags, 1, 1])[indxs,:] 
    # timing
    switches = np.expand_dims(np.concatenate( (np.sum(abs(np.diff(dmat, axis=0)),axis=1), [0]), axis=0), axis=1)
    Xs = NDNutils.create_time_embedding( switches, [num_dlags, 1, 1])[indxs,:]

    tpar = NDNutils.ffnetwork_params( 
        xstim_n=[0], input_dims=[1,1,1, num_dlags], layer_sizes=[1], verbose=False,
        layer_types=['normal'], act_funcs=['lin'], reg_list={'d2t':[None],'l1':[None ]})
    bpar = deepcopy(tpar)
    bpar['xstim_n'] = [1]
    dpar = NDNutils.ffnetwork_params( 
        xstim_n=[2], input_dims=[1,ND2-1,1, num_dlags], layer_sizes=[1], verbose=False,
        layer_types=['normal'], act_funcs=['lin'], reg_list={'d2xt':[None],'l1':[None]})
    comb_parT = NDNutils.ffnetwork_params( 
        xstim_n=None, ffnet_n=[0,1], layer_sizes=[1], verbose=False,
        layer_types=['normal'], act_funcs=['softplus'])
    comb_par = deepcopy(comb_parT)
    comb_par['ffnet_n'] = [0,1,2]

    if spiking:
        nd = 'poisson'
    else:
        nd = 'gaussian'
        
    Tglm = NDN.NDN( [tpar, bpar, comb_parT], noise_dist=nd, tf_seed = 5)
    DTglm = NDN.NDN( [tpar, bpar, dpar, comb_par], noise_dist=nd, tf_seed = 5)
    v2fT = Tglm.fit_variables( layers_to_skip=[2], fit_biases=False)
    v2fT[2][0]['fit_biases'] = True
    v2f = DTglm.fit_variables( layers_to_skip=[3], fit_biases=False)
    v2f[3][0]['fit_biases'] = True

    if (fr1or3 == 3) or (fr1or3 == 1):
        frs_valid = Einfo['frs'] == fr1or3
    else:
        frs_valid = Einfo['frs'] > 0
    to_use = frs_valid[indxs]

    if len(resp) > len(indxs):
        r = deepcopy(resp[indxs])
    else:
        r = deepcopy(resp)
    
    _= Tglm.train(
        input_data=[Xs[to_use,:], Xb[to_use,:]], output_data=r[to_use], learning_alg='lbfgs',# fit_variables=v2fT,
        opt_params=opt_params)
    _= DTglm.train(
        input_data=[Xs[to_use,:], Xb[to_use,:], Xd[to_use,:]], output_data=r[to_use], # fit_variables=v2f, 
        learning_alg='lbfgs', opt_params=opt_params)
    #p1 = Tglm.eval_models(input_data=Xs[indxs,:], output_data=r)[0]
    #p2 = DTglm.eval_models(input_data=[Xs[indxs,:], Xd[indxs,:]], output_data=r)[0]
    #print( "Model performances: %0.4f  -> %0.4f"%(p1, p2) )
    
    # make predictions of each
    predT = Tglm.generate_prediction( input_data=[Xs, Xb] )
    predD = DTglm.generate_prediction( input_data=[Xs, Xb, Xd] )
    
    return predD, predT


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
    Bmatdat = sio.loadmat(datadir+filename)
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

    dispt_raw = NDNutils.shift_mat_zpad( Bmatdat['all_disps'][:,0], time_shift, 0 ) # time shift to keep consistent
    # this has the actual disparity values, which are at the resolution of single bars, and centered around the neurons
    # disparity (sometime shifted to drive neurons well)
    # Sometimes a slightly disparity is used, so it helps to round the values at some resolution
    dispt = np.round(dispt_raw*100)/100
    disp_list = np.unique(dispt)
    # where it is -1009 this corresponds to a blank frame
    # where it is -1005 this corresponds to uncorrelated images between the eyes
    corrt = NDNutils.shift_mat_zpad( Bmatdat['all_corrs'][:,0], time_shift, 0 )
    frs = NDNutils.shift_mat_zpad( Bmatdat['all_frs'][:,0], time_shift, 0 )

    rep_inds = np.add(Bmatdat['rep_inds'][0], -1)  

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
        stim_all: formatted as NT x 72 (stimuli in each eye are cropped to NX=36). It will be time-shifted by 1 
                  to eliminate 0-latency stim. Note this is all stim in experiment, val_inds from used_inds...
        Robs: response concatenating all SUs and MUs, NT x NC. NSU and full Robs is saved as part of Eadd_info.
              This is already selected by used_inds_cell, so no need for further reduction
        used_inds_cell: indices overwhich data is valid for that particular cell. Should be applied to stim only
        UiC, XiC: cross-validation indices for that cell, based on used_inds_cell
        Eadd_info: dictionary containing all other relevant info for experiment
    """

    stim_all, Robs_all, DFs, used_inds, Einfo = binocular_data_import( datadir, expt_num )

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
    Einfo['Robs'] = Robs # all cells, full indexing

    print( "Adjusted for cell %d: %d time points"%(cell_num, NT))

    return stim_all, Robs, used_cell, Ui, Xi, Einfo

