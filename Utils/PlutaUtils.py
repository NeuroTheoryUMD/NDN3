### UTILTIES FOR ANALYZING PLUTA DATA ###
### This is all NDN-3 specific currently
import numpy as np
from copy import deepcopy
import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN


def n2s(n):
    if n < 10:
        s = '0'+str(n)
    else:
        s = str(n)
    return s


def LVsmoother( LVs, boxcarW ):
    if len(LVs.shape)==1:
        LVs = np.expand_dims(LVs, axis=1)
    smLVs = deepcopy(LVs)
    for nn in range(LVs.shape[1]):
        smLVs[:,nn] = np.convolve(LVs[:,nn], np.ones(boxcarW)/boxcarW, mode='same')
    return smLVs


def LdecoderV( LVs, target, tr_inds, te_inds, fit_par=None, verbose=True, skip_pred=False, L2reg=0.001 ):  
    if fit_par is None:
        # Go to default LBFGS
        _, fit_par = optim_params_generic()

    if len(LVs.shape) == 1:
        LVs = np.expand_dims(LVs, axis=1)
    NLVs = LVs.shape[1]
    lin_dec_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NLVs,1], layer_sizes=[1],
        layer_types=['normal'], act_funcs=['lin'], verbose=False,
        reg_list={'l2':[L2reg]} )
    lindec = NDN.NDN( lin_dec_par, noise_dist='gaussian' )
    _ = lindec.train(input_data=LVs, output_data=target, train_indxs=tr_inds, test_indxs=te_inds,
                    #learning_alg='adam', opt_params=adam_params, output_dir=output_dir )
                    learning_alg='lbfgs', opt_params=fit_par )
    mse0 = lindec.eval_models(input_data=LVs[te_inds,:], output_data=target[te_inds])[0]
    mse0x = lindec.eval_models(input_data=LVs[tr_inds,:], output_data=target[tr_inds])[0]
    R2 = 1-mse0/np.var(target[te_inds])
    if verbose:
        print('Train R2 =', 1-mse0x/np.var(target[tr_inds]) )
        print('Test R2 =', R2 )
    if not skip_pred:
        pred = lindec.generate_prediction(input_data=LVs)
        return R2, lindec.copy_model(), pred
    else:
        return R2, lindec.copy_model()
    

def LdecoderNL( LVs, target, tr_inds, te_inds, fit_par=None, verbose=True, skip_pred=False, L2reg=0.001 ):
    if fit_par is None:
        # Go to default LBFGS
        _, fit_par = optim_params_generic()

    if len(LVs.shape) == 1:
        LVs = np.expand_dims(LVs, axis=1)
    NLVs = LVs.shape[1]
    lin_dec_par = NDNutils.ffnetwork_params( 
        input_dims=[1,NLVs,1], layer_sizes=[NLVs//2, 1],
        layer_types=['normal']*2, act_funcs=['relu', 'tanh'], verbose=False,
        reg_list={'l2':[L2reg, L2reg]} )
    lindec = NDN.NDN( lin_dec_par, noise_dist='gaussian' )
    _ = lindec.train(input_data=LVs, output_data=target, train_indxs=tr_inds, test_indxs=te_inds, 
                    #learning_alg='adam', opt_params=fit_par, output_dir=output_dir )
                    learning_alg='lbfgs', opt_params=fit_par )
    mse0 = lindec.eval_models(input_data=LVs[te_inds,:], output_data=target[te_inds])[0]
    mse0x = lindec.eval_models(input_data=LVs[tr_inds,:], output_data=target[tr_inds])[0]
    R2 = 1-mse0/np.var(target[te_inds])
    if verbose:
        print('Train R2 =', 1-mse0x/np.var(target[tr_inds]) )
        print('Test R2 =', R2 )
    if skip_pred:
        return R2, lindec.copy_model()
    else:
        pred = lindec.generate_prediction(input_data=LVs)
        return R2, lindec.copy_model(), pred


def WTAs( Ton, Rs, val_inds, r0=5, r1=30):
    """
    Inputs: 
        Ton: list of touch onsets for all 4 whiskers
        Rs: Robs 
        r0, r1: how many lags before and after touch onset to include (and block out)
    Output:
        wtas: whisker-triggered averages of firing rate
        nontouchFRs: average firing rate (spike prob) away from all four whisker touches
        """
    L = r1+r0
    NT, NC = Rs.shape
    wtas = np.zeros([L,4,NC])
    wcounts = deepcopy(wtas)
    valws = np.zeros([NT,4])
    nontouchFRs = np.zeros([5,NC])
    for ww in range(4):
        valws[val_inds.astype(int),ww] = 1.0
        wts = np.where(Ton[:,ww] > 0)[0]
        print( "w%d: %d touches"%(ww+1, len(wts)))
        for tt in wts:
            t0 = np.maximum(tt-r0,0)
            t1 = np.minimum(tt+r1,NT)
            if t1-t0 == L: # then valid event
                footprint= np.expand_dims(valws[range(t0,t1),ww],1) # All touches probably valid, but just in case
                wcounts[:,ww,:] += footprint
                wtas[:,ww,:] += Rs[range(t0,t1),:]*footprint
            valws[range(t0,t1),ww] = 0
        wtas[:,ww,:] = wtas[:,ww,:] / wcounts[:,ww,:] 
        nontouchFRs[ww,:] = np.sum(valws[:,[ww]]*Rs,axis=0)/np.sum(valws[:,ww])

    # Stats where there are no touches from any whisker
    valtot = np.expand_dims(np.prod(valws, axis=1), 1)
    nontouchFRs[4,:] = np.sum(valtot*Rs,axis=0)/np.sum(valtot)
    
    return wtas, nontouchFRs


def create_NLmap_design_matrix( x, num_bins, val_inds=None, thresh=5, 
                               borderL=None, borderR=None, anchorL=True, rightskip=False):
    """Make design matrix of certain number of bins that maps variable of interest
    anchorL is so there is not an overall bias fit implicitly"""
    NT = x.shape[0]
    if val_inds is None:
        val_inds = range(NT)    
    #m = np.mean(x[val_inds])
    # Determine 5% and 95% intervals (related to thresh)
    h, be = np.histogram(x[val_inds], bins=100)
    h = h/np.sum(h)*100
    cumu = 0
    if borderL is None:
        borderL = np.nan
    if borderR is None:
        borderR = np.nan
    for nn in range(len(h)):
        cumu += h[nn]
        if np.isnan(borderL) and (cumu >=  thresh):
            borderL = be[nn]
        if np.isnan(borderR) and (cumu >= 100-thresh):
            borderR = be[nn]
    # equal divisions between 95-max
    if rightskip:
        bins = np.arange(num_bins)*(borderR-borderL)/num_bins + borderL
    else:
        bins = np.arange(num_bins+1)*(borderR-borderL)/num_bins + borderL
    print(bins)
    XNL = NDNutils.design_matrix_tent_basis( x, bins, zero_left=anchorL )
    return XNL


def extract_time_windows( blks, trange=np.arange(300,1200), XVtr=None ):
    blksA = np.array(blks)
    Nblks = blksA.shape[0]
    Uindx, Xindx = [], []
    if XVtr is None:
        #    Ub = np.arange(blks, dtype='int32')
        XVtr = []
    #else:
        #Xb = np.array(XVtr, dtype='int32')
        #Ub = np.array(list(set(range(Nblks))-set(XVtr)), dtype='int32')
    for tr in range(Nblks):
        ts = trange+blksA[tr,0]
        ts = ts[ts < blksA[tr,1]]
        if tr in XVtr:
            Xindx = np.concatenate( (Xindx, ts), axis=0 )
        else:
            Uindx = np.concatenate( (Uindx, ts), axis=0 )
    return Uindx.astype(int), Xindx.astype(int)


def times_in_trial( blks, tbin, used_trials=None, verbose=False ):
    if used_trials is not None:
        tlist = blks[used_trials, 0] + tbin
    else:
        tlist = blks[:, 0] + tbin
    val_trials = np.where(tlist < blks[:, 1])[0]
    if len(val_trials) < len(tlist):
        if verbose:
            print('  ' +str(len(val_trials))+' trials')
            return tlist(val_trials)
    else:
        return tlist
    
    
def Ldecoder_time_in_trial( LVs, target, blks, T=600, LVsmoothing=None, NLdecoder=False, tskip=2 ):
    
    #NLVs = LVs.shape[1]
    if LVsmoothing is None:
        LVsm = LVs
    else:
        LVsm = LVsmoother(LVs, LVsmoothing)
    R2s = np.zeros(T)
    Ns = np.zeros(T)
    trange = np.arange(0,T,tskip)
    for nn in len(trange):
        tt = trange(nn)
        tindx = times_in_trial( blks, tt )
        Ns[tt] = len(tindx)
        dum=range(int(Ns[tt]))
        if NLdecoder:
            R2s[tt],_ = LdecoderNL( LVsm[tindx,:], target[tindx], dum, dum, skip_pred=True, verbose=False )
        else:
            R2s[tt],_ = LdecoderV( LVsm[tindx,:], target[tindx], dum, dum, skip_pred=True, verbose=False )
        if nn%25 == 0:
            print(tt, R2s[tt])
    return R2s, Ns


def time_range_in_trial( blks, tbins, used_trials=None, verbose=False ):
    if used_trials is None:
        used_trials = range(blks.shape[0])
    tlist = []
    for tr in used_trials:
        tpot = tbins+blks[tr,0]
        tpot = tpot[np.where(tpot < blks[tr,1])[0]]
        tlist = np.concatenate( (tlist, tpot), axis=0 )
    return tlist


def Ldecoder_within_trial( LVs, target, blks, Tmax=1300, twin = None, LVsmoothing=None, NLdecoder=False, verbose=True ):
    
    #NLVs = LVs.shape[1]
    if LVsmoothing is None:
        LVsm = LVs
    else:
        LVsm = LVsmoother(LVs, LVsmoothing)
    
    Tstart = np.arange(0, Tmax, twin, dtype='int64')
    Nwin = len(Tstart)
    R2s = np.zeros(Nwin)
    
    Utr, Xtr = NDNutils.generate_xv_folds(blks.shape[0])
    
    for nn in range(Nwin):
        ts = np.arange(Tstart[nn], Tstart[nn]+twin, dtype='int64')
        TRts = time_range_in_trial( blks, ts, used_trials=Utr )
        TEts = time_range_in_trial( blks, ts, used_trials=Xtr )
        if NLdecoder:
            R2s[nn],_ = LdecoderNL( LVsm, target, TRts.astype(int), TEts, skip_pred=True, verbose=False )
        else:
            R2s[nn],_ = LdecoderV( LVsm, target, TRts.astype(int), TEts.astype(int), skip_pred=True, verbose=False )

        if verbose:
            print(nn, R2s[nn])
    return R2s, Tstart


def find_first_locmin(trace, buf=0, sm=0):
    der = np.diff(trace)
    loc = np.where(np.diff(trace[buf:]) >= 0)[0][0]+buf
    return loc


def prop_distrib(events, prop_name):
    assert prop_name in events[0], 'Invalid property name.'
    distrib = np.zeros(len(events))
    for tt in range(len(events)):
        distrib[tt] = events[tt][prop_name]
    return distrib


###### NDN3 specific ######
def optim_params_generic(early_stopping=100, batch_size=8000, display_interval=100, verbose=False):

    adam_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': True}, learning_alg='adam')

    adam_params['batch_size'] = batch_size
    adam_params['display'] = display_interval
    adam_params['epochs_training'] = early_stopping * 100
    adam_params['run_diagnostics'] = False
    adam_params['epsilon'] = 1e-8
    adam_params['early_stop'] = early_stopping
    adam_params['early_stop_mode'] = 1
    adam_params['data_pipe_type'] = 'data_as_var'
    adam_params['learning_rate'] = 1e-3

    if verbose:
        for d in adam_params:
            print("%20s:\t %s" %(d, adam_params[d]))

    lbfgs_params = NDN.NDN.optimizer_defaults(opt_params={'use_gpu': False, 'display': True}, learning_alg='lbfgs')
    lbfgs_params['maxiter'] = 1000

    return adam_params, lbfgs_params


def trial_plot_grant( Einfo, trial_n, predRS=None, predHT=None, predHM=None, T=1200, fighandle=False ):

    import NDN3.Utils.DanUtils as DU
    import matplotlib.pyplot as plt

    f = DU.subplot_setup(5,2, fighandle=True) 
    tax = 2*np.arange(T)    

    if not isinstance(trial_n, list):
        trial_n = [trial_n]

    for nn in range(len(trial_n)):
        ts = range(Einfo['blks'][trial_n[nn], 0], Einfo['blks'][trial_n[nn], 1])
        ts = ts[:T]
        pistons = interp_pistons(Einfo['piston'][trial_n[nn]])
        outcome = interp_outcome(Einfo['outcome'][trial_n[nn]])
        print('Tr%d: Piston config: '%trial_n[nn], pistons, outcome)

    
        # WHISKER ANGLES
        clrs = 'brmc'
        plt.subplot(5,2,nn+1)
        wlist = []
        for ww in range(4):
            if pistons[ww] > 0:
                plt.plot(tax, Einfo['angles'][ts,ww]*180/np.pi,clrs[ww])
                plt.plot(tax, Einfo['angles'][ts,ww]*180/np.pi,clrs[ww])
                wlist.append(ww)
        plt.xlim([-50, 2*T+50])
    
        # WHISKERS INVOLVED
        plt.subplot(5,2,3+nn)
        for ww in range(4):
            plt.plot(tax, 4-ww+0.8*Einfo['touches'][ts,ww], clrs[ww])
        #plt.plot(1+0.8*Einfo['touches'][ts,wlist[1]], clrs[wlist[1]]) 
        # homotopic touches
        hmtouch = np.multiply(Einfo['touches'][ts,0], Einfo['touches'][ts,2]) + \
                              np.multiply(Einfo['touches'][ts,1], Einfo['touches'][ts,3])
        httouch = np.multiply(Einfo['touches'][ts,0], Einfo['touches'][ts,3]) + \
                              np.multiply(Einfo['touches'][ts,1], Einfo['touches'][ts,2])
        plt.plot(tax, 0.8*httouch, 'g')
        plt.plot(tax, 0.8*hmtouch, 'r--')
        plt.ylim([-0.2, 5])
        plt.xlim([-50, 2*T+50])

        # LICKS
        plt.subplot(5,2,5+nn)
        plt.plot(tax, Einfo['licks'][ts],'k')
        plt.xlim([-50, 2*T+50])
        plt.ylim([-0.1, 1.1])

        # HT/HM pred
        plt.subplot(5,2,7+nn)
        #plt.plot(tax, HTpred[ts],'g')
        #plt.plot(tax, HMpred[ts],'r')
        plt.plot(tax, predHT[ts]-predHM[ts],'g')
        plt.xlim([-50, 2*T+50])
        plt.plot([tax[0], tax[-1]], [0,0],'k')

        # RUN SPEED
        plt.subplot(5,2,9+nn) 
        plt.plot(tax, Einfo['runspeed'][ts],'k--')
        if predRS is not None:
            plt.plot(tax, predRS[ts],'b')
        plt.xlim([-50, 2*T+50])
        
    plt.show()
    if fighandle:
        return f

######## Importing original data (pre-HDF5) ########
def trial_parse( frames ):
    trial_starts = np.where(frames == 1)[0]
    num_trials = len(trial_starts)
    blks = np.zeros([num_trials, 2], dtype='int64')
    for nn in range(num_trials-1):
        blks[nn, :] = [trial_starts[nn], trial_starts[nn+1]]
    blks[-1, :] = [trial_starts[-1], len(frames)]
    return blks


def trial_classify( blks, pistons, outcomes=None, opto=None ):
    """outcomes: 1=hit, 2=miss, 3=false alarm, 4=correct reject"""
    #assert pistons is not None, "pistons cannot be empty"
    Ntr = blks.shape[0]
    #= np.mean(blk_inds, axis=1).astype('int64')
    TRpistons = np.zeros(Ntr, dtype='int64')
    TRoutcomes = np.zeros(Ntr, dtype='int64')
    TRopto = np.zeros(Ntr, dtype='int64')

    for nn in range(Ntr):
        ps = np.median(pistons[range(blks[nn,0], blks[nn,1]),:], axis=0)
        TRpistons[nn] = (ps[0] + 2*ps[1] + 4*ps[2] + 8*ps[3]).astype('int64')
        if outcomes is not None:
            os = np.where(np.median(outcomes[range(blks[nn,0], blks[nn,1]),:], axis=0) > 0)[0]
            if len(os) != 1:
                print("Warning: trial %d had unclear outcome."%nn)
            else:
                TRoutcomes[nn] = os[0]+1
        if opto is not None:
            TRopto[nn] = np.median(opto[range(blks[nn,0], blks[nn,1])], axis=0)

    # Reclassify unilateral (or no) touch trials as Rew = 5
    unilateral = np.where((TRpistons <=2) | (TRpistons == 4) | (TRpistons==8))[0]
    TRoutcomes[unilateral] = 5
    
    if opto is not None:
        return(TRpistons, TRoutcomes, TRopto)    
    else:
        return(TRpistons, TRoutcomes)


def process_locations( clocs ):
    """electrode_info: first column is shank membership, second column is electrode depth"""
    hemis = np.where(clocs[:,0] == 1)[0]
    NC = clocs.shape[0]
    if len(hemis) > 1:
        num_cells = [hemis[1], NC-hemis[1]]
    else:
        num_cells = [NC, 0]
    electrode_info = [None]*2
    for hh in range(len(hemis)):
        if hh == 0:
            crange = range(num_cells[0])
        else:
            crange = range(num_cells[0], NC)
        ei = clocs[crange, :]
        electrode_info[hh] = ei[:, 1:]
    return num_cells, electrode_info


def interp_pistons( pp ):
    pinterp = [pp%2, (pp//2)%2, (pp//4)%2, pp//8] 
    return( pinterp)


def interp_outcome( oo ):
    if oo == 1:
        s = 'hit'
    elif oo == 2:
        s = 'miss'
    elif oo == 3:
        s = 'false alarm'
    else:
        s = 'correct reject'
    return s

def condition_LVs( LVs, valinds=None, mean_subtract=False ):
    """Subtract mean and get variance = 1 for all LVs, for purposes of decoding"""
    T, nLVs = LVs.shape
    if valinds is None:
        valinds = range(T)
    cLVs = np.zeros(LVs.shape)
    for nn in range(nLVs):
        if np.var(LVs[:,nn]) == 0:
            nrm = 1
        else:
            nrm = np.std(LVs[:,nn])
        if mean_subtract:
            mns = np.mean(LVs[:,nn])
        else:
            mns = np.min(LVs[:,nn])
            
        cLVs[:,nn] = (deepcopy(LVs[:,nn])-mns)/nrm
    return cLVs