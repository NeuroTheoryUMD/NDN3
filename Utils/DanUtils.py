"""Neural deep network situtation-specific utils by Dan"""

from __future__ import division
import numpy as np
#import NDN3.NDN as NDN
import NDN3.NDNutils as NDNutils
import matplotlib.pyplot as plt


from copy import deepcopy
from sklearn.preprocessing import normalize as sk_normalize


## GENERAL PLOTTING HACKS ##
def plot_norm( k, ksize = [36,20], cmap='gray', max_val=None):
    if max_val is None:
        max_val = np.max(abs(k))
    if len(k.shape) > 1:  # then already reshaped
        kr = k
    else:
        kr = np.reshape(k, ksize)
    plt.imshow(kr, cmap=cmap, vmin=-max_val, vmax=max_val, aspect='auto')


def subplot_setup(num_rows, num_cols, row_height=2, fighandle=False):
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)
    if fighandle is True:
        return fig

## EXTRACT FILTER FROM NDN -- Various  functions
def tbasis_recover_filters(ndn_mod, ffnet=None):

    if ffnet is None:
        ffnet = 0

    assert np.prod(ndn_mod.networks[ffnet].layers[0].filter_dims[1:]) == 1, 'only works with temporal-only basis'


    if hasattr(ndn_mod.networks[ffnet].layers[0], 'filter_basis') and \
        ndn_mod.networks[ffnet].layers[0].filter_basis is not None:
        tkerns = ndn_mod.networks[ffnet].layers[0].filter_basis@ndn_mod.networks[ffnet].layers[0].weights        
    else:
        tkerns = ndn_mod.networks[ffnet].layers[0].weights
    
    num_lags, num_tkerns = tkerns.shape
    
    if len(ndn_mod.networks[ffnet].layers) == 1:
        non_lag_dims = np.prod(ndn_mod.networks[ffnet+1].layers[0].filter_dims) // num_tkerns
        num_filts = ndn_mod.networks[ffnet+1].layers[0].weights.shape[1]
        ws = np.reshape(ndn_mod.networks[ffnet+1].layers[0].weights, [non_lag_dims, num_tkerns, num_filts])
    else:
        non_lag_dims = np.prod(ndn_mod.networks[ffnet].layers[1].filter_dims) // num_tkerns
        num_filts = ndn_mod.networks[ffnet].layers[1].weights.shape[1]
        ws = np.reshape(ndn_mod.networks[ffnet].layers[1].weights, [non_lag_dims, num_tkerns, num_filts])
    # num_lags = ndn_mod.networks[0].layers[0].num_lags
    ks = np.reshape(np.matmul(tkerns, ws), [non_lag_dims*num_lags, num_filts])
    # ks = np.zeros([non_lag_dims*num_lags, num_filts])

    return ks


def compute_spatiotemporal_filters(ndn_mod, ffnet=None):

    if ffnet is None:
        ffnet = 0

    # Check to see if there is a temporal layer first
    num_lags = ndn_mod.networks[ffnet].layers[0].num_lags
    if num_lags == 1 and ndn_mod.networks[ffnet].layers[0].filter_dims[0] > 1:  # if num_lags was set as dim0
        num_lags = ndn_mod.networks[ffnet].layers[0].filter_dims[0]
    if (np.prod(ndn_mod.networks[ffnet].layers[0].filter_dims[1:]) == 1) and \
        (ndn_mod.network_list[ffnet]['layer_types'][0] != 'sep'):  # then likely temporal basis
        ks_flat = tbasis_recover_filters(ndn_mod, ffnet=ffnet)
        if len(ndn_mod.networks[ffnet].layers) > 1:
            sp_dims = ndn_mod.networks[ffnet].layers[1].filter_dims[1:]
            other_dims = ndn_mod.networks[ffnet].layers[1].filter_dims[0] // ndn_mod.networks[ffnet].layers[0].num_filters
        else:
            sp_dims = ndn_mod.networks[ffnet+1].layers[0].filter_dims[1:]
            other_dims = ndn_mod.networks[ffnet+1].layers[0].filter_dims[0] // ndn_mod.networks[ffnet].layers[0].num_filters
    else:
        # Check if separable layer
        n0 = ndn_mod.networks[ffnet].layers[0].input_dims[0] 
        if ndn_mod.network_list[ffnet]['layer_types'][0] == 'sep':
            NF = ndn_mod.networks[ffnet].layers[0].weights.shape[1]
            kt = np.expand_dims(ndn_mod.networks[ffnet].layers[0].weights[range(n0), :].T, 1)
            ksp = np.expand_dims(ndn_mod.networks[ffnet].layers[0].weights[n0:, :].T, 2)
            ks_flat = np.reshape(deepcopy(ksp) @ deepcopy(kt), 
                        [NF, np.prod(ndn_mod.networks[ffnet].layers[0].input_dims)]).T 
            sp_dims = ndn_mod.networks[ffnet].layers[0].input_dims[1:3]
            other_dims = n0 // num_lags
 
        else: # not separable layer and no temporal basis: easiest ca
            ks_flat = ndn_mod.networks[ffnet].layers[0].weights
            sp_dims = ndn_mod.networks[ffnet].layers[0].filter_dims[1:3]
            other_dims = n0 // num_lags
    num_filters = ks_flat.shape[-1]

    # Reshape filters with other_dims tucked into first spatial dimension (on outside)
    if (sp_dims[1] == 1) or (sp_dims[0] == 1):
        ks = np.reshape(np.transpose(
            np.reshape(ks_flat, [np.prod(sp_dims), other_dims, num_lags, num_filters]), [1, 0, 2, 3]),
            [np.prod(sp_dims)*other_dims, num_lags, num_filters])
    else:
        ks = np.reshape(np.transpose(
            np.reshape(ks_flat, [sp_dims[1], sp_dims[0], other_dims, num_lags, num_filters]), [2, 0, 1, 3, 4]),
            [sp_dims[1], sp_dims[0]*other_dims, num_lags, num_filters])

    return ks
# END compute_spatiotemporal_filters


def plot_filters(ndn_mod=None, filters=None, filter_dims=None, tbasis_select=-1, flipxy=False, ffnet=None, cmap=None):
    """Throw in NDN_mod to do defaults. Can also through in ks directly, but must be either 2-d (weights x filter)
    with filter_dims provided, or 3-d (num_lags, num_space, num_filt)"""

    if ffnet is None:
        ffnet = 0
    # Set colormap
    if cmap is None or cmap == 0:
        cmap = 'Greys'
    elif cmap == 1:
        cmap = 'bwr'
    elif cmap == 2:
        cmap = 'RdBu_r'
    # otherwise cmap must be string that works...

    temporal_basis_present = False
    if ndn_mod is None:
        assert filters is not None, 'Must supply filters or ndn_mod'
        num_filters = filters.shape[-1]
        if filter_dims is not None:
            ks = np.reshape(filters, [filter_dims[1], filter_dims[0], num_filters])
        else:
            assert len(filters.shape) == 3, 'filter dims must be provided or ks must be reshaped into 3d'
            ks = filters
    else:
        ks = compute_spatiotemporal_filters(ndn_mod=ndn_mod, ffnet=ffnet)
        if (np.prod(ndn_mod.networks[ffnet].layers[0].filter_dims[1:]) == 1) and \
                 (ndn_mod.network_list[ffnet]['layer_types'][0] != 'sep'):
            temporal_basis_present = True

    if len(ks.shape) > 3:
        # Must remake into 2-d filters
        num_lags, num_filters = ks.shape[2:]
        if num_lags == 1:
            ks = ks[:, :, 0, :]  # removes lag dimension
        else:
            if tbasis_select < 0:
                ks = np.sum(ks, axis=2)
            else:
                ks = ks[:, :, tbasis_select, :]
    else:
        num_filters = ks.shape[2]

    if temporal_basis_present:
        if ndn_mod.networks[0].layers[0].filter_basis is None:
            tkerns = ndn_mod.networks[ffnet].layers[0].weights
        else:
            tkerns = ndn_mod.networks[ffnet].layers[0].filter_basis@ndn_mod.networks[ffnet].layers[0].weights
        plt.plot(tkerns)
        plt.title('Temporal bases')

    if num_filters > 200:
        print('Limiting display to first 200 filters')
        num_filters = 200

    if num_filters/10 == num_filters//10:
        cols = 10
    elif num_filters / 8 == num_filters // 8:
        cols = 8
    elif num_filters / 6 == num_filters // 6:
        cols = 6
    elif num_filters / 5 == num_filters // 5:
        cols = 5
    elif num_filters < 10:
        cols = num_filters
    else:
        cols = 8
    rows = int(np.ceil(num_filters/cols))

    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    fig.set_size_inches(18 / 6 * cols, 7 / 4 * rows)
    plt_scale = np.max(abs(ks))

    for nn in range(num_filters):
        ax = plt.subplot(rows, cols, nn + 1)
        if flipxy:
            k = ks[:, :, nn]
        else:
            k = np.transpose(ks[:, :, nn])
        if k.shape[0] == k.shape[1]:
            plt.imshow(k, cmap=cmap, interpolation='none', vmin=-plt_scale, vmax=plt_scale, aspect=1)
        else:
            plt.imshow(k, cmap=cmap, interpolation='none', vmin=-plt_scale, vmax=plt_scale, aspect='auto')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()
# END plot_filters


def plot_3dfilters(ndnmod=None, filters=None, dims=None, plot_power=False, ffnet=0):

    if ndnmod is None:
        if dims is None:
            assert len(filters.shape) == 4, 'must include filter dims or reshape the input.'
            dims = filters.shape[:3]
            NK = filters.shape[-1]
            ks = np.reshape(deepcopy(filters), [np.prod(dims), NK])
        else:
            NK = filters.shape[-1]
            ks = np.reshape(deepcopy(filters), [np.prod(dims), NK])
    else:
        filters = compute_spatiotemporal_filters(ndnmod, ffnet=ffnet)
        dims = filters.shape[:3]
        NK = filters.shape[-1]
        ks = np.reshape(deepcopy(filters), [np.prod(dims), NK])

    ncol = np.minimum(8, 2*NK)
    nrow = np.ceil(2 * NK / ncol).astype(int)
    subplot_setup(nrow, ncol)
    for nn in range(NK):
        ktmp = np.reshape(ks[:, nn], [dims[0] * dims[1], dims[2]])
        tpower = np.std(ktmp, axis=0)
        bestlag = np.argmax(abs(tpower))
        # Calculate temporal kernels based on max at best-lag
        bestpospix = np.argmax(ktmp[:, bestlag])
        bestnegpix = np.argmin(ktmp[:, bestlag])

        ksp = np.reshape(ks[:, nn], dims)[:, :, bestlag]
        ax = plt.subplot(nrow, ncol, 2*nn+1)
        plt.plot([0, len(tpower)-1], [0, 0], 'k')
        if plot_power:
            plt.plot(tpower, 'b')
            plt.plot(tpower, 'b.')
            plt.plot([bestlag, bestlag], [np.minimum(np.min(kt), 0)*1.1, np.max(kt)*1.1], 'r--')
        else:
            plt.plot(ktmp[bestpospix, :], 'b')
            plt.plot(ktmp[bestpospix, :], 'b.')
            plt.plot(ktmp[bestnegpix, :], 'r')
            plt.plot(ktmp[bestnegpix, :], 'r.')
            minplot = np.minimum(np.min(ktmp[bestnegpix, :]), np.min(ktmp[bestpospix, :]))
            maxplot = np.maximum(np.max(ktmp[bestnegpix, :]), np.max(ktmp[bestpospix, :]))
            plt.plot([bestlag, bestlag], [minplot*1.1, maxplot*1.1], 'k--')
        plt.axis('tight')
        plt.title('c' + str(nn))
        ax.set_xticks([])
        ax.set_yticks([])
        ax = plt.subplot(nrow, ncol, 2*nn+2)
        plt.imshow(ksp, interpolation='none', cmap='Greys', vmin=-np.max(abs(ks[:, nn])), vmax=np.max(abs(ks[:, nn])))
        plt.title('lag=' + str(bestlag))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
# END plot_3dfilters


def plot_scatter( xs, ys, clr='g' ):
    assert len(xs) == len(ys), 'data dont match'
    for nn in range(len(xs)):
        plt.plot([xs[nn]], [ys[nn]], clr+'o', fillstyle='full')
        if clr != 'k':
            plt.plot([xs[nn]], [ys[nn]], 'ko', fillstyle='none')
    #plt.show()
    

def plot_internal_weights(ws, num_inh=None):
    ws_play = deepcopy(ws)
    num_dim = ws.shape[0]
    if num_inh is not None:
        ws_play[range(num_dim-num_inh, num_dim), :] *= -1
    m = np.max(abs(ws_play))
    plt.imshow(ws_play, cmap='bwr', vmin=-m, vmax=m)


## ADDITIONAL REGULARIZATION FUNCTIONS
def unit_reg_test(
    ndn_mod=None, input_data=None, output_data=None, train_indxs=None, test_indxs=None, data_filters=None, 
    reg_type=None, ffnet_targets=[0], layer_targets=[0], reg_vals=None,
    fit_variables=None, opt_params=None, learning_alg='lbfgs', to_plot=True ):

    if reg_vals is None:
        reg_vals = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    assert ndn_mod is not None, 'Must include model.'
    
    if not isinstance(ffnet_targets, list):
        ffnet_targets = [ffnet_targets]
    if not isinstance(layer_targets, list):
        layer_targets = [layer_targets]
        
    Nreg = len(reg_vals)
    NC = output_data.shape[1]
    
    LLmat = np.zeros([Nreg, NC])
    reg_mods = []
    for nn in range(Nreg):
        ndn_iter = ndn_mod.copy_model()
        
        for aa in range(len(ffnet_targets)):
            for bb in range(len(layer_targets)):
                ndn_iter.set_regularization(
                    reg_type, reg_vals[nn], 
                    ffnet_target=ffnet_targets[aa], layer_target=layer_targets[bb])
                
        _=ndn_iter.train( 
            input_data=input_data, output_data=output_data, train_indxs=train_indxs, test_indxs=test_indxs, 
            data_filters=data_filters, fit_variables=fit_variables,
            opt_params=opt_params, learning_alg=learning_alg, silent=True)
        LLmat[nn, :] = ndn_iter.eval_models(
            input_data=input_data, output_data=output_data, data_indxs=test_indxs, data_filters=data_filters)
        reg_mods.append(ndn_iter.copy_model())
        print(nn, reg_vals[nn], np.mean(LLmat[nn, :]))
    
    # Pick optimal vals
    opt_pos = np.argmin(LLmat,axis=0)
    opt_vals = np.zeros(NC)
    for cc in range(NC):
        opt_vals[cc] = reg_vals[opt_pos[cc]]
        
    if to_plot:
        num_rows = NC//8 + 1
        if NC < 8:
            num_col = NC
        else:
            num_col = 8
        subplot_setup(num_rows, num_col)
        for cc in range(NC):
            plt.subplot(num_rows, num_col, cc+1)
            plt.plot(LLmat[:,cc],'b')
            plt.plot(LLmat[:,cc],'b.')
            plt.plot([opt_pos[cc], opt_pos[cc]], plt.ylim(),'k--')
            plt.title('cell '+str(cc+1))
        plt.show()

    reg_results = {
        'reg_type': reg_type,
        'rvals': reg_vals,
        'opt_vals': opt_vals,
        'targets': [ffnet_targets, layer_targets],
        'LLmat': LLmat, 
        'reg_models': reg_mods}
    return reg_results


def unit_assign_reg( ndn_mod, reg_results ):
    
    new_ndn = ndn_mod.copy_model()
    Ntar = reg_results['targets'][0]
    Ltar = reg_results['targets'][1]
    rtype = reg_results['reg_type']
    for aa in Ntar:
        for bb in Ltar:
            new_ndn.networks[aa].layers[bb].convert_to_unit_regularization()
            new_ndn.networks[aa].layers[bb].reg.vals[rtype] = deepcopy(reg_results['opt_vals'])
    return new_ndn


def plot_2dweights( w, input_dims=None, num_inh=0):
    """w can be one dimension (in which case input_dims is required) or already shaped. """

    if input_dims is not None:
        w = np.reshape(w, [input_dims[1], input_dims[0]])
    else:
        input_dims = [w.shape[1], w.shape[0]]

    num_exc = input_dims[0]-num_inh
    w = np.divide(w, np.max(np.abs(w)))

    fig, ax = plt.subplots(1)

    if num_inh > 0:
        w[:, num_exc:] = np.multiply(w[:, num_exc:], -1)
        plt.imshow(w, interpolation='none', cmap='bwr', vmin=-1, vmax=1)
    else:
        plt.imshow(w, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
    if num_inh > 0:
        plt.plot(np.multiply([1, 1], num_exc-0.5), [-0.5, input_dims[1]-0.5], 'k')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


## SPATIAL MEASUREMENT FUNCTIONS
def spatial_profile_info(xprofile):
    """Calculate the mean and standard deviation of xprofile along one dimension"""
    # Calculate mean of filter

    if isinstance(xprofile, list):
        k = np.square(np.array(xprofile))
    else:
        k = np.square(deepcopy(xprofile))

    NX = xprofile.shape[0]

    nrms = np.maximum(np.sum(k), 1e-10)
    mn_pos = np.divide(np.sum(np.multiply(k, range(NX))), nrms)
    xs = np.array([range(NX)] * np.ones([NX, 1])) - np.array([mn_pos] * np.ones([NX, 1]))
    stdev = np.sqrt(np.divide(np.sum(np.multiply(k, np.square(xs))), nrms))
    return mn_pos, stdev
# END spatial_profile_info


def spatial_spread(filters, axis=0):
    """Calculate the spatial spread of a list of filters along one dimension"""

    # Calculate mean of filter
    k = np.square(deepcopy(filters))
    if axis > 0:
        k = np.transpose(k)
    NX, NF = filters.shape

    nrms = np.maximum(np.sum(k,axis=0), 1e-10)
    mn_pos = np.divide(np.sum(np.multiply(np.transpose(k), range(NX)), axis=1), nrms)
    xs = np.array([range(NX)] * np.ones([NF, 1])) - np.transpose(np.array([mn_pos] * np.ones([NX, 1])))
    stdevs = np.sqrt(np.divide(np.sum(np.multiply(np.transpose(k), np.square(xs)), axis=1), nrms))

    return stdevs
# END spatial_spread


## SIDE/SCAFFOLD NETWORK ANALYSIS to understand solutions
def side_network_analyze(side_ndn, cell_to_plot=None, plot_aspect='auto'):
    """
    Applies to NDN with a side network (conv or non-conv. It will divide up the weights
    of the side-network into layer-specific pieces and resize to represent space and
    filter number as different inputs.

    Inputs:
        side_ndn: the network model (required)
        cell_to_plot: if plots desired, single out the cell to plot (default no plot). If
            convolutional, then can only plot specified cell. If non-convolutonal, then
            just set to something other than 'None', and will plot weights for all cells.
        plot_aspect: if plots, then whether to have aspect as 'auto' (default) or 'equal'
    Output:
        returns the weights as organized as descrived above
    """

    if plot_aspect != 'auto':
        plot_aspect = 'equal'

    # Check to see if NSM is convolutional or normal
    if (side_ndn.network_list[0]['layer_types'][0] == 'conv') or \
            (side_ndn.network_list[0]['layer_types'][0] == 'biconv'):
        is_conv = True
    else:
        is_conv = False

    # number of spatial dims depends on whether first side layer is convolutional or not
    #num_space = side_ndn.networks[0].input_dims[1]*side_ndn.networks[0].input_dims[2]
    num_space = int(side_ndn.networks[1].layers[0].weights.shape[0]/ side_ndn.networks[1].input_dims[0])
    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]
    filter_nums = side_ndn.networks[1].num_units[:]
    #filter_nums = side_ndn.network_list[0]['layer_sizes'][:]
    num_layers = len(filter_nums)

    # Adjust effective space/filter number if binocular model
    if side_ndn.network_list[0]['layer_types'][0] == 'biconv':
        num_space = num_space // 2
        filter_nums[0] *= 2

    if cell_to_plot is not None:
        fig, ax = plt.subplots(nrows=1, ncols=num_layers)
        fig.set_size_inches(16, 3)

    # Reshape whole weight matrix
    wside = np.reshape(side_ndn.networks[1].layers[0].weights, [num_space, np.sum(filter_nums), num_cells])
    num_inh = side_ndn.network_list[0]['num_inh']

    # identify max and min weights for plotting (if plotting)
    if cell_to_plot is not None:
        if is_conv:
            # find normalization of layers for relevant cell
            img_max = np.max(wside[:, cell_to_plot])
            img_min = np.min(wside[:, cell_to_plot])
        else:
            img_max = np.max(wside)
            img_min = np.min(wside)
        # equalize scaling around zero
        if img_max > -img_min:
            img_min = -img_max
        else:
            img_max = -img_min

    fcount = 0
    ws = []
    for ll in range(num_layers):
        #wtemp = wside[range(ll, len(wside), num_layers), :]
        wtemp = wside[:, range(fcount, fcount+filter_nums[ll]), :]
        ws.append(deepcopy(wtemp))
        fcount += filter_nums[ll]

        if cell_to_plot is not None:
            plt.subplot(1, num_layers, ll+1)
            if is_conv:
                plt.imshow(wtemp[:, :, cell_to_plot], aspect=plot_aspect, interpolation='none', cmap='bwr',
                           vmin=img_min, vmax=img_max)
                # Put line in for inhibitory units
                if num_inh[ll] > 0:
                    plt.plot(np.multiply([1, 1], filter_nums[ll]-num_inh[ll]-0.5), [-0.5, num_space-0.5], 'r')
                # Put line in for binocular layer (if applicable)
                if side_ndn.network_list[0]['layer_types'][ll] == 'biconv':
                    plt.plot([filter_nums[ll]/2, filter_nums[ll]/2], [-0.5, num_space-0.5], 'w')
            else:
                plt.imshow(np.transpose(wtemp), aspect='auto', interpolation='none', cmap='bwr',
                           vmin=img_min, vmax=img_max)  # will plot all cells
                # Put line in for inhibitory units
                if num_inh[ll] > 0:
                    plt.plot(np.multiply([1, 1], filter_nums[ll]-num_inh[ll]-0.5), [-0.5, num_cells-0.5], 'r')

    plt.show()
    return ws


def side_network_properties(side_ndn, norm_type=0):
    """Returns measurements of scaffold weights as they are distributed across layer"""

    ws = side_network_analyze(side_ndn)
    wside = side_ndn.networks[-1].layers[-1].weights
    cell_nrms = np.sqrt(np.sum(np.square(wside), axis=0))
    NC = len(cell_nrms)
    num_layers = len(ws)
    NX = side_ndn.network_list[0]['input_dims'][1]

    if (side_ndn.network_list[0]['layer_types'][0] == 'conv') or (side_ndn.network_list[0]['layer_types'][0] == 'biconv'):
        conv_net = True
    else:
        conv_net = False
    assert conv_net is True, 'Convolutional network only for now.'

    # Calculate layer weights
    num_inh = side_ndn.network_list[0]['num_inh']
    layer_weights = np.zeros([num_layers, NC], dtype='float32')
    spatial_weights = np.zeros([num_layers, NX, NC], dtype='float32')
    EIlayer = np.zeros([2, num_layers, NC], dtype='float32')
    EIspatial = np.zeros([2, num_layers, NX, NC], dtype='float32')
    for ll in range(num_layers):
        if conv_net:
            layer_weights[ll, :] = np.sqrt(np.sum(np.sum(np.square(ws[ll]), axis=0), axis=0)) / cell_nrms
            spatial_weights[ll, :, :] = np.divide(np.sqrt(np.sum(np.square(ws[ll]), axis=1)), cell_nrms)
            if num_inh[ll] > 0:
                NE = ws[ll].shape[1] - num_inh[ll]
                elocs = range(NE)
                ilocs = range(NE, ws[ll].shape[1])
                if norm_type == 0:
                    EIspatial[0, ll, :, :] = np.sum(ws[ll][:, elocs, :], axis=1)
                    EIspatial[1, ll, :, :] = np.sum(ws[ll][:, ilocs, :], axis=1)
                    EIlayer[:, ll, :] = np.sum(EIspatial[:, ll, :, :], axis=1)
                else:
                    EIspatial[0, ll, :, :] = np.sqrt(np.sum(np.square(ws[ll][:, elocs, :]), axis=1))
                    EIspatial[1, ll, :, :] = np.sqrt(np.sum(np.square(ws[ll][:, ilocs, :]), axis=1))
                    EIlayer[:, ll, :] = np.sqrt(np.sum(np.square(EIspatial[:, ll, :, :]), axis=1))

        else:
            layer_weights[ll, :] = np.sqrt(np.sum(np.square(ws[ll]), axis=0)) / cell_nrms

    if np.sum(num_inh) > 0:
        if norm_type == 0:
            Enorm = np.sum(EIlayer[0, :, :], axis=0)
        else:
            Enorm = np.sqrt(np.sum(np.square(EIlayer[0, :, :]), axis=0))

        EIlayer = np.divide(EIlayer, Enorm)
        EIspatial = np.divide(EIspatial, Enorm)

    props = {'layer_weights': layer_weights,
             'spatial_profile': spatial_weights,
             'EIspatial': EIspatial,
             'EIlayer': EIlayer}

    return props


def scaffold_similarity(side_ndn, c1, c2, level=None, EI=None):
    """Similarity between scaffold vectors of two cells.
    Assume network is convolutional -- otherwise wont make sense"""

    ws = side_network_analyze(side_ndn)
    NX, NC = ws[0].shape[0], ws[0].shape[2]
    assert (c1 < NC) and (c2 < NC), 'cells out of range'

    num_inh = side_ndn.network_list[0]['num_inh']
    NUs = side_ndn.network_list[0]['layer_sizes']
    if np.sum(num_inh) == 0:
        EI = None

    if level is not None:
        assert level < len(ws), 'level too large'
        if EI is not None:
            if EI > 0:  # then excitatory only
                w1 = ws[level][:, range(NUs[level]-num_inh[level]), c1]
                w2 = ws[level][:, range(NUs[level]-num_inh[level]), c2]
            else:  # then inhibitory only
                w1 = ws[level][:, range(NUs[level]-num_inh[level], NUs[level]), c1]
                w2 = ws[level][:, range(NUs[level]-num_inh[level], NUs[level]), c2]
        else:
            w1 = ws[level][:, :, c1]
            w2 = ws[level][:, :, c2]
    else:
        if EI is not None:
            if EI > 0:  # then excitatory only
                w1 = ws[0][:, range(NUs[0]-num_inh[0]), c1]
                w2 = ws[0][:, range(NUs[0]-num_inh[0]), c2]
                for ll in range(1, len(ws)):
                    w1 = np.concatenate((w1, ws[ll][:, range(NUs[ll]-num_inh[ll]), c1]), axis=1)
                    w2 = np.concatenate((w2, ws[ll][:, range(NUs[ll]-num_inh[ll]), c2]), axis=1)
            else:
                w1 = ws[0][:, range(NUs[0]-num_inh[0], NUs[0]), c1]
                w2 = ws[0][:, range(NUs[0]-num_inh[0], NUs[0]), c2]
                for ll in range(1, len(ws)):
                    w1 = np.concatenate((w1, ws[ll][:, range(NUs[ll]-num_inh[ll], NUs[ll]), c1]), axis=1)
                    w2 = np.concatenate((w2, ws[ll][:, range(NUs[ll]-num_inh[ll], NUs[ll]), c2]), axis=1)
        else:
            w1 = ws[0][:, :, c1]
            w2 = ws[0][:, :, c2]
            for ll in range(1, len(ws)):
                w1 = np.concatenate((w1, ws[ll][:, :, c1]), axis=1)
                w2 = np.concatenate((w2, ws[ll][:, :, c2]), axis=1)

    # Normalize
    nrm1 = np.sqrt(np.sum(np.square(w1)))
    nrm2 = np.sqrt(np.sum(np.square(w2)))
    if (nrm1 == 0) or (nrm2 == 0):
        return 0.0
    w1 = np.divide(w1, nrm1)
    w2 = np.divide(w2, nrm2)

    # Shift w2 to have highest overlap with w1
    ds = np.zeros(2*NX-1)
    for sh in range(2*NX-1):
        ds[sh] = np.sum(np.multiply(w1, NDNutils.shift_mat_zpad(w2, sh-NX+1, dim=0)))

    return np.max(ds)


def scaffold_similarity_vector(side_ndn, c1, level=None, EI=None):
    """compares distances between one cell and all others"""
    NC = side_ndn.networks[-1].layers[-1].weights.shape[1]
    dvec = np.zeros(NC, dtype='float32')
    for cc in range(NC):
        dvec[cc] = scaffold_similarity(side_ndn, c1, cc, level=level, EI=EI)
    dvec[c1] = 1

    return dvec


def scaffold_similarity_matrix(side_ndn, level=None, EI=None):

    NC = side_ndn.networks[-1].layers[-1].weights.shape[1]
    dmat = np.ones([NC, NC], dtype='float32')
    for c1 in range(NC):
        for c2 in range(c1+1,NC):
            dmat[c1, c2] = scaffold_similarity(side_ndn, c1, c2, level=level, EI=EI)
            dmat[c2, c1] = dmat[c1, c2]
    return dmat


## SCAFFOLD AND FF-NETWORK ANALYSIS TO REVISE and MANIPULATE MODELS
def scaffold_density(side_ndn, internal_ws=True):
    """Plots the total use of all units within the scaffold"""

    num_units = side_ndn.network_list[1]['layer_sizes']
    num_inh = side_ndn.network_list[1]['num_inh']
    cfws = side_ndn.network_list[1]['conv_filter_widths']
    tes = np.maximum(side_ndn.network_list[1]['time_expand'], 1)
    num_lvls = len(num_units)
    NU = np.sum(num_units)
    #print(NU, side_ndn.networks[2].layers[0].weights.shape[0]//NX)
    # Normalized scaffold structure -- each cell gets same number of votes -- all positive
    ws = np.reshape(
        np.divide(side_ndn.networks[2].layers[0].weights, np.sum(side_ndn.networks[2].layers[0].weights,axis=0)),
        [NX, NU, NCtot])
    sc_density = np.sum(ws, axis=2)
    scd_pm = deepcopy(sc_density)
    #plt.imshow(sc_density, cmap='YlOrRd')
    for nn in range(len(num_units)):
        barrier = np.sum(num_units[:(nn+1)])
        scd_pm[:,range(barrier-num_inh[nn], barrier)] *= -1

    num_rows = 2    
    subplot_setup(num_rows, num_lvls)
    
    for nn in range(num_lvls):
        rng = range(np.sum(num_units[:(nn)]).astype(int), np.sum(num_units[:(nn+1)]).astype(int))
        ax=plt.subplot(num_rows, num_lvls, nn+1)
        plt.imshow(scd_pm[:, rng], cmap='bwr',vmin=-np.max(sc_density), vmax=np.max(sc_density), aspect='auto')
        plt.plot(np.multiply([1,1], num_units[nn]-num_inh[nn]), [0, NX-1], 'k')
        ax.set_xticklabels([])
        plt.title("Lvl %d"%(nn+1))
        
        plt.subplot(num_rows, num_lvls, nn+num_lvls+1)
        plt.plot(np.sum(sc_density[:,rng], axis=0), 'k')
        maxw = np.max(np.sum(sc_density,axis=0))
        if internal_ws and nn < (num_lvls-1):
            int_ws = deepcopy(side_ndn.networks[1].layers[nn+1].weights)
            int_ws = np.sum(np.sum(
                np.reshape(
                    deepcopy(side_ndn.networks[1].layers[nn+1].weights), 
                    [cfws[nn+1]*tes[nn+1], num_units[nn], num_units[nn+1]]), 
                axis=2), axis=0)
            maxw2 = np.max(int_ws)
            plt.plot(int_ws/maxw2*maxw, 'g')
        plt.plot(np.multiply([1,1], num_units[nn]-num_inh[nn]), [0, maxw*1.1], 'r--')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.autoscale(enable=True, axis='y', tight=True)
    plt.show()


def evaluate_ffnetwork_units(ffnet, end_weighting=None, to_plot=False, thresh_list=None, percent_drop=None):
    """Analyze FFnetwork nodes to determine their contribution in the big picture.
    thresh_list and percent_drop apply criteria for each layer (one or other) to suggest units to drop"""

    num_layers = len(ffnet.layers)
    num_unit_bot = ffnet.layers[-1].weights.shape[1]
    if end_weighting is None:
        prev_ws = np.ones(num_unit_bot, dtype='float32')
    else:
        assert len(end_weighting) == num_unit_bot, 'end_weighting has wrong dimensionality'
        prev_ws = end_weighting
    # Process prev_ws: nothing less than zeros, and sum to 1
    prev_ws = np.maximum(prev_ws, 0)
    prev_ws = np.divide(prev_ws, np.mean(prev_ws))

    node_eval = [[]]*num_layers
    node_eval[-1] = deepcopy(prev_ws)
    for nn in range(num_layers-1):
        ws = deepcopy(ffnet.layers[num_layers-1-nn].weights)
        next_ws = np.matmul(np.square(ws), prev_ws)
        node_eval[num_layers-nn-2] = np.divide(deepcopy(next_ws), np.mean(next_ws))
        prev_ws = deepcopy(next_ws)

    # Determine units to drop (if any)
    units_to_drop = [[]]*num_layers
    remaining_units = [[]] * num_layers
    if percent_drop is not None:
        # overwrite thresh_list
        thresh_list = [0]*num_layers
        if len(percent_drop) == 1:
            percent_drop = percent_drop*num_layers
        for nn in range(num_layers):
            num_units = len(node_eval[nn])
            unit_order = np.argsort(node_eval[nn])
            cutoff = np.maximum(int(np.floor(num_units * percent_drop[nn])-1), 0)
            if cutoff == num_units - 1:
                remaining_units[nn] = range(num_units)
                thresh_list[nn] = 0
            else:
                remaining_units[nn] = unit_order[range(cutoff, num_units)]
                units_to_drop[nn] = unit_order[range(cutoff)]
                thresh_list[nn] = np.mean([node_eval[nn][unit_order[cutoff]], node_eval[nn][unit_order[cutoff + 1]]])
    else:
        if thresh_list is None:
            thresh_list = [None]*num_layers
        else:
            if thresh_list is not list:
                TypeError('thresh_list must be list.')
        if len(thresh_list) == 1:
            thresh_list = [thresh_list]*num_layers
        for nn in range(num_layers):
            if thresh_list[nn] is None:
                thresh_list[nn] = 0.2 * np.max(node_eval[nn])
            remaining_units[nn] = np.where(node_eval[nn] >= thresh_list[nn])[0]
            units_to_drop[nn] = np.where(node_eval[nn] < thresh_list[nn])[0]
            if len(remaining_units[nn]) == 0:
                print('layer %d: threshold too high' % nn)

    if to_plot:
        subplot_setup( num_rows=1, num_cols=num_layers)
        for nn in range(num_layers):
            plt.subplot(1, num_layers, nn+1)
            plt.plot(node_eval[nn], 'b')
            plt.plot(node_eval[nn], 'b.')
            NF = node_eval[nn].shape[0]
            plt.plot([0, NF-1], [thresh_list[nn], thresh_list[nn]], 'r')
            plt.xlim([0, NF-1])
        plt.show()

    return node_eval, remaining_units, units_to_drop


def tunnel_fit(ndn_mod, end_weighting=None, thresh_list=None, percent_drop=None):
    """Set up model with weights reset and coupled"""

    assert end_weighting is not None, 'Must supply end_weighting for this to work.'

    node_eval, good_nodes, tunnel_units = \
        evaluate_ffnetwork_units(ndn_mod.networks[0], end_weighting=end_weighting,
                           thresh_list=thresh_list, percent_drop=percent_drop)
    num_layers = len(node_eval)
    ndn_copy = ndn_mod.copy_model()

    # First randomize below-threshold filters in first level
    num_stix = ndn_copy.networks[0].layers[0].weights.shape[0]
    ndn_copy.networks[0].layers[0].weights[:, tunnel_units[0]] = \
        np.random.normal(size=[num_stix, len(tunnel_units[0])], scale=1/np.sqrt(num_stix))
    # Connect with rest of tunnel (and dissociate from rest of network
    for nn in range(1, num_layers):
        # Detach ok_units from previous-layer bad units
        #ok_units = list(set(range(len(node_eval[nn])))-set(tunnel_units[nn]))
        for mm in good_nodes[nn]:
            ndn_copy.networks[0].layers[nn].weights[tunnel_units[nn-1], mm] = 0
        for mm in tunnel_units[nn]:
            ndn_copy.networks[0].layers[nn].weights[:, mm] = np.zeros([len(node_eval[nn-1])], dtype='float32')
            ndn_copy.networks[0].layers[nn].weights[tunnel_units[nn-1], mm] = \
                np.random.normal(size=[len(tunnel_units[nn-1])], scale=1/np.sqrt(len(tunnel_units[nn-1])))

    return ndn_copy


def ffnet_health(ndn_mod, toplot=True):

    num_nets = len(ndn_mod.networks)
    whealth = [None]*num_nets
    bhealth = [None]*num_nets
    max_layers = 0
    for nn in range(num_nets):
        num_layers = len(ndn_mod.networks[nn].layers)
        whealth[nn] = [None]*num_layers
        bhealth[nn] = [None]*num_layers
        if num_layers > max_layers:
            max_layers = num_layers
        for ll in range(num_layers):
            whealth[nn][ll] = np.std(ndn_mod.networks[nn].layers[ll].weights, axis=0)
            bhealth[nn][ll] = ndn_mod.networks[nn].layers[ll].biases[0, :]

    if toplot:
        subplot_setup(num_nets * 2, max_layers + 1)
        for nn in range(num_nets):
            num_layers = len(ndn_mod.networks[nn].layers)
            for ll in range(num_layers):
                plt.subplot(num_nets*2, max_layers, (2*nn)*max_layers+ll+1)
                _=plt.hist(whealth[nn][ll], 20)
                plt.title("n%dL%d ws" % (nn, ll))
                plt.subplot(num_nets*2, max_layers, (2*nn+1)*max_layers+ll+1)
                _=plt.hist(bhealth[nn][ll], 20)
                plt.title("n%dL%d bs" % (nn, ll))
        plt.show()

    return whealth, bhealth


def prune_ndn(ndn_mod, end_weighting=None, thresh_list=None, percent_drop=None):
    """Remove below-threshold nodes of network. Set thresholds to 0 if don't want to touch layer
        Also should not prune last layer (Robs), but can for multi-networks
        BUT CURRENTLY ONLY WORKS WITH SINGLE-NETWORK NDNs"""

    node_eval, remaining_units, _ = \
        evaluate_ffnetwork_units(ndn_mod.networks[0], end_weighting=end_weighting,
                           thresh_list=thresh_list, percent_drop=percent_drop)
    num_layers = len(node_eval)

    net_lists = deepcopy(ndn_mod.network_list)
    layer_sizes = net_lists[0]['layer_sizes']
    num_inh = net_lists[0]['num_inh']
    for nn in range(num_layers):
        if num_inh[nn] > 0:
            # update number of inhibitory units based on how many left
            num_inh[nn] = np.where(remaining_units[nn] > (layer_sizes[nn]-num_inh[nn]))[0].shape[0]
        layer_sizes[nn] = len(remaining_units[nn])

    net_lists[0]['layer_sizes'] = layer_sizes
    net_lists[0]['num_inh'] = num_inh

    # Make new NDN
    pruned_ndn = NDN.NDN(net_lists, noise_dist=ndn_mod.noise_dist, ffnet_out=ndn_mod.ffnet_out, tf_seed=ndn_mod.tf_seed)
    # Copy all the relevant weights and stuff
    for net_n in range(len(net_lists)):
        if net_n == 0:
            pruned_ndn.networks[0].layers[0].weights = \
                ndn_mod.networks[0].layers[0].weights[:, remaining_units[0]].copy()
            pruned_ndn.networks[0].layers[0].biases[0, :] = \
                ndn_mod.networks[0].layers[0].biases[0, remaining_units[0]].copy()
        else:
            pruned_ndn.networks[0].layers[0].weights = ndn_mod.networks[0].layers[0].weights.copy()
            pruned_ndn.networks[0].layers[0].biases = ndn_mod.networks[0].layers[0].biases.copy()

        for nn in range(1, len(pruned_ndn.networks[0].layers)):
            if net_n == 0:
                for mm in range(len(remaining_units[nn])):
                    cc = remaining_units[nn][mm]
                    pruned_ndn.networks[net_n].layers[nn].weights[:, mm] = \
                        ndn_mod.networks[net_n].layers[nn].weights[remaining_units[nn-1], cc].copy()
                    pruned_ndn.networks[net_n].layers[nn].biases[0, mm] = \
                        ndn_mod.networks[net_n].layers[nn].biases[0, cc]
            else:
                pruned_ndn.networks[net_n].layers[nn].weights = ndn_mod.networks[net_n].layers[nn].weights.copy()
                pruned_ndn.networks[net_n].layers[nn].biases = ndn_mod.networks[net_n].layers[nn].biases.copy()

    return pruned_ndn


def train_bottom_units( ndn_mod=None, unit_eval=None, num_units=None,
                        input_data=None, output_data=None, train_indxs=None, test_indxs=None,
                        data_filters=None, opt_params=None):

    MIN_UNITS = 10

    if ndn_mod is None:
        TypeError('Must define ndn_mod')
    if unit_eval is None:
        TypeError('Must input unit_eval')
    if input_data is None:
        TypeError('Forgot input_data')
    if output_data is None:
        TypeError('Forgot output_data')
    if train_indxs is None:
        TypeError('Forgot train_indxs')

    # Make new NDN
    netlist = deepcopy(ndn_mod.network_list)
    layer_sizes = netlist[0]['layer_sizes']
    num_units_full = layer_sizes[-1]

    # Default train bottom 20% of units
    if num_units is None:
        num_units = int(np.floor(num_units_full*0.2))
    size_ratio = num_units/num_units_full
    for nn in range(len(layer_sizes)-1):
        layer_sizes[nn] = int(np.maximum(size_ratio*layer_sizes[nn], MIN_UNITS))
        netlist[0]['num_inh'][nn] = int(np.floor(size_ratio*netlist[0]['num_inh'][nn]))
        netlist[0]['weights_initializers'][nn] = 'trunc_normal'
    layer_sizes[-1] = num_units
    netlist[0]['layer_sizes'] = layer_sizes  # might not be necessary because python is dumb
    netlist[0]['weights_initializers'][-1] = 'trunc_normal'
    small_ndn = NDN.NDN(netlist, noise_dist=ndn_mod.noise_dist)
    sorted_units = np.argsort(unit_eval)
    selected_units = sorted_units[range(num_units)]
    # Adjust Robs
    robs_small = output_data[:, selected_units]
    if data_filters is not None:
        data_filters_small = data_filters[:, selected_units]
    else:
        data_filters_small = None

    # Train
    _= small_ndn.train(input_data=input_data, output_data=robs_small, data_filters=data_filters_small,
                       learning_alg='adam', train_indxs=train_indxs, test_indxs=test_indxs, opt_params=opt_params)
    LLs = small_ndn.eval_models(input_data=input_data, output_data=robs_small,
                                data_indxs=test_indxs, data_filters=data_filters_small)

    return small_ndn, LLs, selected_units


def join_ndns(ndn1, ndn2, units2=None):
    """Puts all layers from both ndns into 1, except the last [output] layer, which is inherited
    from ndn1 only. However, new units of ndn2 (earlier layers will be connected """
    num_net = len(ndn1.networks)
    num_net2 = len(ndn2.networks)
    assert num_net == num_net2, 'Network number does not match'

    new_netlist = deepcopy(ndn1.network_list)
    num_units = [[]]*num_net
    for nn in range(num_net):
        num_layers = len(ndn1.networks[nn].layers)
        num_layers2 = len(ndn2.networks[nn].layers)
        assert num_layers == num_layers2, 'Layer number does not match'
        layer_sizes = [0]*num_layers
        num_units[nn] = [[]]*num_layers
        if ndn1.ffnet_out[0] == -1:
            ndn1.ffnet_out[0] = len(ndn1.networks)
        for ll in range(num_layers):
            if (nn != ndn1.ffnet_out[0]) or (ll < num_layers-1):
                num_units[nn][ll] = [ndn1.networks[nn].layers[ll].weights.shape[1],
                                     ndn2.networks[nn].layers[ll].weights.shape[1]]
            else:
                num_units[nn][ll] = [ndn1.networks[nn].layers[ll].weights.shape[1], 0]
            layer_sizes[ll] = np.sum(num_units[nn][ll])
            num_units[nn][ll][1] = layer_sizes[ll]  # need the sum anyway (see below)
            new_netlist[nn]['weights_initializers'][ll] = 'zeros'

        new_netlist[nn]['layer_sizes'] = layer_sizes

    joint_ndn = NDN.NDN(new_netlist, noise_dist=ndn1.noise_dist)
    # Assign weights and biases
    for nn in range(num_net):
        # First layer simple since input has not changed
        ll = 0
        joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0])] = \
            ndn1.networks[nn].layers[ll].weights
        joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] = \
            ndn2.networks[nn].layers[ll].weights
        joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0])] = \
            ndn1.networks[nn].layers[ll].biases
        joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] = \
            ndn2.networks[nn].layers[ll].biases

        for ll in range(1, len(num_units[nn])):
            joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0])] =\
                ndn1.networks[nn].layers[ll].biases.copy()
            weight_strip = np.zeros([num_units[nn][ll-1][1], num_units[nn][ll][0]], dtype='float32')
            weight_strip[range(num_units[nn][ll-1][0]), :] = ndn1.networks[nn].layers[ll].weights.copy()
            joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0])] = weight_strip
            if (nn != ndn1.ffnet_out[0]) or (ll < num_layers-1):
                weight_strip = np.zeros([num_units[nn][ll - 1][1],
                                         num_units[nn][ll][1] - num_units[nn][ll][0]], dtype='float32')
                weight_strip[range(num_units[nn][ll - 1][0], num_units[nn][ll - 1][1]), :] = \
                    ndn2.networks[nn].layers[ll].weights.copy()
                joint_ndn.networks[nn].layers[ll].weights[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] =\
                    weight_strip
                joint_ndn.networks[nn].layers[ll].biases[:, range(num_units[nn][ll][0], num_units[nn][ll][1])] =\
                    ndn2.networks[nn].layers[ll].biases.copy()
            elif units2 is not None:
                weight_strip = np.zeros([num_units[nn][ll - 1][1], len(units2)], dtype='float32')
                weight_strip[range(num_units[nn][ll - 1][0], num_units[nn][ll - 1][1]), :] = \
                    ndn2.networks[nn].layers[ll].weights.copy()
                joint_ndn.networks[nn].layers[ll].weights[:, units2] = weight_strip

    return joint_ndn


## SCAFFOLD DISPLAY FUNCTIONS
def side_ei_analyze(side_ndn):

    num_space = int(side_ndn.networks[0].input_dims[1])
    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]

    num_units = side_ndn.networks[1].num_units[:]
    num_layers = len(num_units)

    wside = np.reshape(side_ndn.networks[1].layers[0].weights, [num_space, np.sum(num_units), num_cells])
    num_inh = side_ndn.network_list[0]['num_inh']

    cell_nrms = np.sum(side_ndn.networks[1].layers[0].weights, axis=0)
    tlayer_present = side_ndn.networks[0].layers[0].filter_dims[1] == 1
    if tlayer_present:
        nlayers = num_layers - 1
    else:
        nlayers = num_layers
    # EIweights = np.zeros([2, nlayers, num_cells])
    EIprofiles = np.zeros([2, nlayers, num_space, num_cells])
    num_exc = np.subtract(num_units, num_inh)

    fcount = 0
    for ll in range(num_layers):
        ws = deepcopy(wside[:, range(fcount, fcount+num_units[ll]), :])
        fcount += num_units[ll]
        if num_inh[ll] == 0:
            ews = np.maximum(ws, 0)
            iws = np.minimum(ws, 0)
        else:
            ews = ws[:, range(num_exc[ll]), :]
            iws = ws[:, range(num_exc[ll], num_units[ll]), :]
        if (ll == 0) or (tlayer_present == False):
            EIprofiles[0, ll, :, :] = np.divide( np.sum(ews, axis=1), cell_nrms )
            EIprofiles[1, ll, :, :] = np.divide( np.sum(iws, axis=1), cell_nrms )
        if tlayer_present:
            EIprofiles[0, ll-1, :, :] += np.divide( np.sum(ews, axis=1), cell_nrms )
            EIprofiles[1, ll-1, :, :] += np.divide( np.sum(iws, axis=1), cell_nrms )
    EIweights = np.sum(EIprofiles, axis=2)

    return EIweights, EIprofiles


def scaffold_nonconv_plot( side_ndn, with_inh=True, nolabels=True, skip_first_level=False, linewidth=1):

    # validity check
    assert len(side_ndn.network_list) == 2, 'This does not seem to be a standard scaffold network.'

    num_cells = side_ndn.network_list[1]['layer_sizes'][-1]
    num_units = side_ndn.networks[1].num_units[:]
    num_layers = len(num_units)
    scaff_ws = side_ndn.networks[1].layers[0].weights
    cell_nrms = np.max(np.abs(side_ndn.networks[1].layers[0].weights), axis=0)
    num_inh = side_ndn.network_list[0]['num_inh']
    num_exc = np.subtract(num_units, num_inh)

    fcount = 0
    col_mod = 0
    if skip_first_level:
        col_mod = 1

    subplot_setup(num_rows=1, num_cols=num_layers-col_mod)
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.linewidth'] = linewidth
    for ll in range(num_layers):
        ws = np.transpose(np.divide(deepcopy(scaff_ws[range(fcount, fcount+num_units[ll]), :]), cell_nrms))

        fcount += num_units[ll]
        if (num_inh[ll] > 0) and with_inh:
            ws[:, num_exc[ll]:] = np.multiply( ws[:, num_exc[ll]:], -1)
        if not skip_first_level or (ll > 0):
            ax = plt.subplot(1, num_layers-col_mod, ll+1-col_mod)

            if with_inh:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='bwr', vmin=-1, vmax=1)
            else:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
            if num_inh[ll] > 0:
                plt.plot(np.multiply([1, 1], num_exc[ll]-0.5), [-0.5, num_cells-0.5], 'k')
            if ~nolabels:
                ax.set_xticks([])
                ax.set_yticks([])
    plt.show()


def scaffold_plot_cell(side_ndn, cell_n, with_inh=True, nolabels=True, skip_first_level=False, linewidth=1):
    """Plots the scaffold weight vector for a convolutional network for the chosen neuron"""

    # either way assume scaffold is last later, and see where it maps from
    sc_tar = side_ndn.network_list[-1]['ffnet_n'][0]
    assert sc_tar is not None, 'Unable to read scaffold target.'
    #if len(side_ndn.network_list) != 2:
    #    print('Non-typical scaffold network: scaffold = network %d, target = %d' %(len(side_ndn.network_list)-1, sc_tar) )

    num_cells = side_ndn.network_list[-1]['layer_sizes'][-1]
    num_units = side_ndn.networks[-1].num_units[:]
    num_layers = len(num_units)

    if 'biconv' in side_ndn.network_list[sc_tar]['layer_types']:
        num_space = int(side_ndn.networks[sc_tar].input_dims[1])//2
    else:
        num_space = int(side_ndn.networks[sc_tar].input_dims[1])

    cell_nrms = np.max(np.abs(side_ndn.networks[-1].layers[0].weights), axis=0)
    wside = np.reshape(side_ndn.networks[-1].layers[0].weights, [num_space, np.sum(num_units), num_cells])
    num_inh = side_ndn.network_list[sc_tar]['num_inh']
    num_exc = np.subtract(num_units, num_inh)
    #cell_nrms = np.sum(side_ndn.networks[1].layers[0].weights, axis=0)

    fcount = 0
    col_mod = 0
    if skip_first_level:
        col_mod = 1

    subplot_setup(num_rows=1, num_cols=num_layers-col_mod)
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['axes.linewidth'] = linewidth
    for ll in range(num_layers):
        ws = np.divide(deepcopy(wside[:, range(fcount, fcount+num_units[ll]), cell_n]), cell_nrms[cell_n])

        fcount += num_units[ll]
        if (num_inh[ll] > 0) and with_inh:
            ws[:, num_exc[ll]:] = np.multiply( ws[:, num_exc[ll]:], -1)
            if side_ndn.network_list[sc_tar]['layer_types'][ll] == 'biconv':
                more_inh_range = range(num_units[ll]//2-num_inh[ll], num_units[ll]//2)
                ws[:, more_inh_range] = np.multiply(ws[:, more_inh_range], -1)
        if not skip_first_level or (ll > 0):
            ax = plt.subplot(1, num_layers-col_mod, ll+1-col_mod)

            if with_inh:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='bwr', vmin=-1, vmax=1)
            else:
                plt.imshow(ws, aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=1)
            if side_ndn.network_list[sc_tar]['layer_types'][ll] == 'biconv':
                plt.plot(np.multiply([1, 1], num_units[ll]//2 - 0.5), [-0.5, num_space - 0.5], 'k')
            if num_inh[ll] > 0:
                plt.plot(np.multiply([1, 1], num_exc[ll]-0.5), [-0.5, num_space-0.5], 'k')
                if side_ndn.network_list[sc_tar]['layer_types'][ll] == 'biconv':
                    plt.plot(np.multiply([1, 1], num_exc[ll]-num_units[ll]//2 - 0.5), [-0.5, num_space - 0.5], 'k')
            if ~nolabels:
                ax.set_xticks([])
                ax.set_yticks([])
    plt.show()


## RANDOM UTILITY FUNCTIONS
def filename_num2str( n, num_digits=2 ):
    num_places = int(np.ceil(np.log10(n)+0.001))
    place_shift = int(np.maximum(num_digits-num_places, 0))
    s = '0'*place_shift + str(n%(10**num_digits))[:]
    return s


def threeDmax(k):
    a,b,c = k.shape
    cbest = np.argmax( np.max(np.reshape(k, [a*b, c]), axis=0) )
    bbest = np.argmax( np.max(k[:,:,cbest], axis=0) )
    abest = np.argmax( k[:,bbest, cbest] )
    return abest, bbest, cbest
    
    
def figure_export( fig_handle, filename, bitmap=False, dpi=300):
    """Usage: figure_export( fig_handle, filename, variable_list, bitmap=False, dpi=300)
    if bitmap, will use dpi and export as .png. Otherwise will export PDF"""

    if bitmap:
        fig_handle.savefig( filename, bbox_inches='tight', dpi=dpi, transparent=True )
    else:
        fig_handle.savefig( filename, bbox_inches='tight', transparent=True )


def figure_format(
    num_rows=1, num_cols=1, row_height=None, col_width=None, squares=False, 
    xticks=True, yticks=True, xticklabels=True, yticklabels=True,
    fig_handle = False):
    """Format figure (and return figure handle) using default/desired appearance/size
    Optional Input Arguments (all default values): 
        num_rows [1]: number of rows in figure (assuming subplots, unless 1)
        num_cols [1]: number of columns in figure
        col_width: width of each column (in inches) 
            [Default: if num_cols = 1, then 6. Otherwise divides up full width (16) between rows] 
        row_height: height of row (in inches)
            [Default: if num_rows = 1, then 4. Otherwise, 2/3 of col_width]
        squares [False]: If set to true, makes all subplots the same horizontal/vertical
            dimension, picking the minimum of row_height and col_width
        xticks, yticks [True]: can turn off x-ticks or y-ticks by setting to False. Can also
            directly make an array/list to explicitly set where ticks are
        xticklabels, yticklabels [True]: same rules about xticks/yticks, applied to tick-labels
        fig_handle [False]: whether to return a figure_handle
    """
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    plt.tight_layout()
    # Assume normal plot if num_rows_cols = 1 (6x4)
    if col_width is None:
        if num_cols == 1:
            col_width = 6
        else:
            col_width = 16/num_cols
    else:
        col_width = np.minimum(16/num_cols, col_width)  # large number brings full column width
    if row_height is None:
        if num_rows == 1:
            row_height = 4
        else:
            row_height = col_width*2/3
    if squares:
        col_width = np.minimum(col_width, row_height)
        row_height = col_width
    fig.set_size_inches(col_width*num_cols, row_height*num_rows) 

    # Axes
    if num_rows*num_cols == 1:
        ax = [ax]
    for nn in range(len(ax)):
        ax[nn].spines['top'].set_visible(False)
        ax[nn].spines['right'].set_visible(False)
        if type(xticks) == bool:
            if not xticks:
                ax[nn].get_xaxis().set_ticks([])
        else:
            ax[nn].get_xaxis().set_ticks(xticks)
        if type(yticks) == bool:
            if not yticks:
                ax[nn].get_yaxis().set_ticks([])
        else:
            ax[nn].get_yaxis().set_ticks(yticks)

        if type(xticklabels) == bool:
            if not xticklabels:
                ax[nn].get_xaxis().set_ticklabels([])
        else:
            ax[nn].get_xaxis().set_ticklabels(xticklabels)
        if type(yticklabels) == bool:
            if not yticklabels:
                ax[nn].get_yaxis().set_ticklabels([])
        else:
            ax[nn].get_yaxis().set_ticklabels(yticks)

    if fig_handle:
        return fig


def matlab_export(filename, variable_list):
    """Export list of variables to .mat file"""

    import scipy.io as sio
    if not isinstance(variable_list, list):
        variable_list = [variable_list]

    matdata = {}
    for nn in range(len(variable_list)):
        assert not isinstance(variable_list[nn], list), 'Cant do a list of lists.'
        if nn < 10:
            key_name = 'v0' + str(nn)
        else:
            key_name = 'v' + str(nn)
        matdata[key_name] = variable_list[nn]

    sio.savemat(filename, matdata)


def save_python_data( filename, data ):
    with open( filename, 'wb') as f:
        np.save(f, data)
    print( 'Saved data to', filename )


def load_python_data( filename, show_keys=False ):

    with open( filename, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    print( 'Loaded data from', filename )
    if len(data.shape) == 0:
        data = data.flat[0]  # to rescue dictionaries
    if (type(data) is dict) and show_keys:
        print(data.keys())
    return data


def display_matrix( x, prec=3, spacing=4, number_rows=False, number_cols=False ):
    a, b = x.shape
    s = "  %" + str(spacing+prec) + "." + str(prec) + "f"
    if number_cols:
        if number_rows:
            print( "    ", end='')
        for mm in range(b):
            print( " "*(spacing+prec-2)+ "[%2d]"%mm, end='' )
        print('')
    for nn in range(a):
        if number_rows:
            print( "[%2d]"%nn, end='' )
        for mm in range(b):
            print( s%x[nn, mm], end='')
        print('')

        
def entropy(dist):

    # normalize distribution
    dist = np.divide( dist.astype('float32'), np.sum(dist) )
    # make all zeros 1
    dist[np.where(dist == 0)[0]] = 1
    H = -np.sum( np.multiply( dist, np.log2(dist)) )

    return H


def orthogonalize_gram_schmidt(kmat):
    """Orthogonalize filters using gram_schmidt method. kmat should be num_params x num_filts"""

    num_par, num_filts = kmat.shape
    # First normalize all filters
    kmat_out = sk_normalize(kmat, axis=0)

    for nn in range(num_filts-1):

        # orthogonalize all filters to the chosen
        for mm in range(nn+1, num_filts):
            kmat_out[:, mm] = kmat_out[:, mm] - np.dot(kmat_out[:, nn], kmat_out[:, mm]) * kmat_out[:, nn]

        # renoramlize
        kmat_out = sk_normalize(kmat_out, axis=0)

    return kmat_out


def best_val_mat(mat, min_or_max=0):

    # Make sure a matrix (numpy)
    mat = np.array(mat, dtype='float32')
    if min_or_max == 0:
        ax0 = np.min(mat, axis=0)
        b1 = np.argmin(ax0)
        b0 = np.argmin(mat[:, b1])
    else:
        ax0 = np.max(mat, axis=0)
        b1 = np.argmax(ax0)
        b0 = np.argmax(mat[:, b1])

    b0 = int(b0)
    b1 = int(b1)
    return b0, b1


def gabor_sized(dim, angle, phase_select=0):
    k = np.zeros([2 * dim + 1, 2 * dim + 1], dtype='float32')
    a1, a2 = np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)
    sigma = dim / np.sqrt(6)
    omega = 1.5 * np.pi / dim
    for xx in range(-dim, dim + 1):
        for yy in range(-dim, dim + 1):
            if phase_select == 0:
                k[xx + dim, yy + dim] = np.cos(omega * (a1 * xx + a2 * yy)) * np.exp(
                    -(xx * xx + yy * yy) / (2 * sigma * sigma))
            else:
                k[xx + dim, yy + dim] = np.sin(omega * (a1 * xx + a2 * yy)) * np.exp(
                    -(xx * xx + yy * yy) / (2 * sigma * sigma))
    return k


def gabor_array(dim, num_angles=6, both_phases=False):
    """Make array of Gabors, sized by gabor_sized (above), which preserves one full phase within a circular
    aperture sized relative to dim. The length of a size is 2*dim+1, centered in the middle, so this returns
    an array of gabors with dimensions (2*dim+1)^2 x num_angles. By default it only returnes cosine gabors, but
    making both_phases=True gives double the Gabors, with the sines following cosines."""

    L = 2*dim+1
    if both_phases:
        gabors = np.zeros([L*L, num_angles, 2], dtype='float32')
        num_gabors = 2*num_angles
    else:
        num_gabors = num_angles
        gabors = np.zeros([L*L, num_angles, 1], dtype='float32')
    #gabors = np.zeros([L*L, num_gabors], dtype='float32')
    for nn in range(num_angles):
        gabors[:, nn, 0] = np.reshape(gabor_sized(dim, nn*180/num_angles, 0), [L*L])
        if both_phases:
            gabors[:, nn, 1] = np.reshape(gabor_sized(dim, nn*180/num_angles, 1), [L*L])
    
    return np.reshape(gabors, [L*L, num_gabors])


def design_matrix_drift( NT, anchors, zero_left=True, zero_right=False, const_right=False, to_plot=False):
    """Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
    Here s is a continuous variable (e.g., a stimulus) that is function of time -- single dimension --
    and this will generate apply a tent basis set to s with a basis variable for each anchor point. 
    The end anchor points will be one-sided, but these can be dropped by changing "zero_left" and/or
    "zero_right" into "True".

    Inputs: 
        NT: length of design matrix
        anchors: list or array of anchor points for tent-basis set
        zero_left, zero_right: boolean whether to drop the edge bases (default for both is False)
    Outputs:
        X: design matrix that will be NT x the number of anchors left after zeroing out left and right
    """

    if const_right:
        zero_right = True

    anchors = list(anchors)
    if anchors[0] > 0:
        anchors = [0] + anchors
    if anchors[-1] < NT:
        anchors = anchors + [NT]
    NA = len(anchors)

    X = np.zeros([NT, NA])
    for aa in range(NA):
        if aa > 0:
            dx = anchors[aa]-anchors[aa-1]
            X[range(anchors[aa-1], anchors[aa]), aa] = np.arange(dx)/dx
        if aa < NA-1:
            dx = anchors[aa+1]-anchors[aa]
            X[range(anchors[aa], anchors[aa+1]), aa] = 1-np.arange(dx)/dx

    if zero_left:
        X = X[:, 1:]
    if zero_right:
        X = X[:, :-1]
        if const_right:
            X[range(anchors[-2], anchors[-1]), -1] = 1.0

    if to_plot:
        plt.imshow(X.T, aspect='auto', interpolation='none')
        plt.show()

    return X
