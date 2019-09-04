"""Basic temporal-layer definitions"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from regularization import Regularization
from regularization import SepRegularization

from layer import Layer


# TODO: fix input/output dims so that multiple consecutive TLayers are possible
class TLayer(Layer):
    """Implementation of a temporal layer

    Attributes:
        filter_width (int): time spread
        batch_size (int): the batch size is explicitly needed for this computation

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            num_lags=None,
            num_filters=None,
            activation_func='lin',
            dilation=1,
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):

        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): size (and dimensionality) of the inputs to the layer. Should be in the
                form of: [# temporal lags, # space dim1, # space dim2, other dimensions]
                Thus, must be list: length 1 means temporal only, length 2 means one time and one space, etc
                For TLayer: the number of lags should be 1, since they will be generated internally though num_lags.
                    Altenatively, this will be moved over in constructor if num_lags = None
                Default is none, which means will be determined at the build time
            num_lags (int): number of lags to be generated within the TLayer (and operated on by filters)
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        #self.batch_size = batch_size
        #self.filter_width = filter_width

        assert num_lags > 1, 'time_expand must be greater than 1 if using a temporal layer'
        # Handle more than lags in dimension-0 if necessary
        if input_dims[0]//num_lags > 1:
            input_dims = [num_lags, input_dims[1], input_dims[2], input_dims[0]//num_lags]

        # If output dimensions already established, just strip out num_filters
        #  if isinstance(num_filters, list):
        #      num_filters = num_filters[0]

        super(TLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            filter_dims=[num_lags, 1, 1],
            output_dims=num_filters,  # Note difference from layer¸
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        self.num_lags = num_lags
        self.input_dims[0] = self.input_dims[0]//num_lags
        # self.output_dims = deepcopy(input_dims)
        #self.output_dims[0] = self.num_filters # ORIGINAL
        self.output_dims = input_dims.copy()
        self.output_dims += [num_filters]
        #self.internal_dims = [1, num_filters]  # removed lags, now just filters

        self.dilation = dilation

        # ei_mask not useful at the moment
        self.ei_mask_var = None

        self.reg = Regularization(
            #input_dims=[filter_width, 1, 1],
            input_dims=[num_lags, 1, 1],
            num_outputs=num_filters,
            vals=reg_initializer)

    # END TLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        #assert batch_size is not None, "must pass in batch_size to TLayer"
        num_inputs = self.input_dims[1]*self.input_dims[2]
        if len(self.internal_dims) > 1:
            num_inputs *= np.prod(self.internal_dims[1:])

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs), [num_inputs, batch_size, 1, 1])

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            #padding = tf.constant([[0, self.filter_width], [0, 0]])
            padding = tf.constant([[0, self.num_lags], [0, 0]])
            padded_filt = tf.pad(tf.reverse(w_pn, [0]), padding)  # note the time-reversal
            #shaped_padded_filt = tf.reshape(padded_filt, [2*self.filter_width, 1, 1, self.num_filters])
            shaped_padded_filt = tf.reshape(padded_filt, [2*self.num_lags, 1, 1, self.num_filters])

            # convolve
            strides = [1, 1, 1, 1]
            dilations = [1, self.dilation, 1, 1]
            _pre = tf.nn.conv2d(shaped_input, shaped_padded_filt, strides, dilations=dilations, padding='SAME')
            pre = tf.add(_pre, self.biases_var)
            post = self._apply_act_func(pre)

            self.outputs = tf.reshape(tf.transpose(post, [1, 0, 2, 3]), (batch_size, -1))

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END TLayer.build_graph


class CaTentLayer(Layer):
    """Implementation of calcium tent layer

    Attributes:
        filter_width (int): time spread
        batch_size (int): the batch size is explicitly needed for this computation

    """

    def __init__(
            self,
            scope=None,
            nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            num_filters=None,
            filter_width=None,  # this can be a list up to 3-dimensions
            batch_size=None,
            activation_func='lin',
            normalize_weights=True,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=True,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        self.batch_size = batch_size
        self.filter_width = filter_width

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional (space)
            input_dims = [1, input_dims, 1]

        # If output dimensions already established, just strip out num_filters
        #  if isinstance(num_filters, list):
        #      num_filters = num_filters[0]

        # TODO: how to specify num filters...
        if num_filters > 1:
            num_filters = input_dims[1]

        super(CaTentLayer, self).__init__(
            scope=scope,
            #nlags=nlags,
            input_dims=input_dims,
            output_dims=output_dims,  # Note difference from layer
            #my_num_inputs=filter_width,
            #my_num_outputs=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        self.output_dims = input_dims

        # ei_mask not useful at the moment
        self.ei_mask_var = None

        # make nc biases instead of only 1
        bias_dims = (1, self.output_dims[1])
        init_biases = np.zeros(shape=bias_dims, dtype='float32')
        self.biases = init_biases

        self.reg = Regularization(
            input_dims=[filter_width, 1, 1],
            num_outputs=num_filters,
            vals=reg_initializer)

    # END CaTentLayer.__init__

    def build_graph(self, inputs, params_dict=None, use_dropout=False):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs), [-1, self.batch_size, 1, 1])

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            padding = tf.constant([[0, self.filter_width], [0, 0]])
            padded_filt = tf.pad(w_pn, padding)
            shaped_padded_filt = tf.reshape(padded_filt, [2*self.filter_width, 1, 1, self.num_filters])

            # convolve
            strides = [1, 1, 1, 1]
            _pre = tf.nn.conv2d(shaped_input, shaped_padded_filt, strides, padding='SAME')

            # transpose, squeeze to final shape
            # both cases will produce _pre_final_shape.shape ---> (batch_size, nc)
            if self.num_filters > 1:
                _pre_final_shape = tf.linalg.diag_part(tf.transpose(tf.squeeze(_pre, axis=2), [1, 0, 2]))
            else:
                # single filter
                _pre_final_shape = tf.transpose(tf.squeeze(_pre, axis=[2, 3]))

            pre = tf.add(_pre_final_shape, self.biases_var)
            self.outputs = self._apply_act_func(pre)

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END CaTentLayer.build_graph


class NoRollCaTentLayer(Layer):
    """Implementation of calcium tent layer

    Attributes:
        filter_width (int): time spread
        batch_size (int): the batch size is explicitly needed for this computation

    """

    def __init__(
            self,
            scope=None,
            nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            num_filters=None,
            filter_width=None,  # this can be a list up to 3-dimensions
            batch_size=None,
            activation_func='lin',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=True,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        if filter_width is None:
            filter_width = 2*batch_size

        self.batch_size = batch_size
        self.filter_width = filter_width

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional (space)
            input_dims = [1, input_dims, 1]

        # If output dimensions already established, just strip out num_filters
      #  if isinstance(num_filters, list):
      #      num_filters = num_filters[0]

        # TODO: how to specify num filters...
        if num_filters > 1:
            num_filters = input_dims[1]

        super(NoRollCaTentLayer, self).__init__(
            scope=scope,
            nlags=nlags,
            input_dims=input_dims,
            output_dims=output_dims,  # Note difference from layer
            my_num_inputs=filter_width,
            my_num_outputs=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        self.output_dims = input_dims

    # END CaTentLayer.__init__

    def build_graph(self, inputs, params_dict=None):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs), [self.input_dims[1], self.batch_size, 1, 1])

            # make shaped filt
            conv_filt_shape = [self.filter_width, 1, 1, self.num_filters]

            if self.normalize_weights > 0:
                wnorms = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(self.weights_var), axis=0)), 1e-8)
                shaped_filt = tf.reshape(tf.divide(self.weights_var, wnorms), conv_filt_shape)
            else:
                shaped_filt = tf.reshape(self.weights_var, conv_filt_shape)

            # convolve
            strides = [1, 1, 1, 1]
            if self.pos_constraint:
                pre = tf.nn.conv2d(shaped_input, tf.maximum(0.0, shaped_filt), strides, padding='SAME')
            else:
                pre = tf.nn.conv2d(shaped_input, shaped_filt, strides, padding='SAME')

            # from pre to post
            if self.ei_mask_var is not None:
                post = tf.multiply(
                    self.activation_func(tf.add(pre, self.biases_var)),
                    self.ei_mask_var)
            else:
                post = self.activation_func(tf.add(pre, self.biases_var))

            # this produces shape (batch_size, nc, num_filts)
            # after matrix_diag_part we have diagonal part ---> shape will be (batch_size, nc)
            self.outputs = tf.matrix_diag_part(tf.transpose(tf.squeeze(post, axis=2), [1, 0, 2]))

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END NoRollCaTentLayer.build_graph
