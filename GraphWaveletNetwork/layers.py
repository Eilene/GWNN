from inits import *
import tensorflow as tf
from sklearn.preprocessing import normalize
flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                # self.vars['weights_diffusion_'+str(i)] = tf.Variable(1.0)
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            # support = dot(self.support[i], self.vars['weights_diffusion_'+str(i)] * pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolution_WeightShare(Layer):
    """Graph convolution layer."""
    def __init__(self, weight_normalize,input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_WeightShare, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.weight_normalize = weight_normalize
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))
            for i in range(len(self.support)):
                self.vars['weights_diffusion_'+str(i)] = tf.Variable(1.0)

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(0)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(0)]
            # support = dot(self.support[i], pre_sup, sparse=True)
            support = dot( self.support[i], self.vars['weights_diffusion_'+str(i)] *pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # # supports = list()
        # output = self.vars['weights_diffusion_'+str(0)] *  self.support[0]
        # output = None
        # for i in range( len( self.support)):
        #     # print(self.support[i])
        #     # support = self.vars['weights_diffusion_'+str(i)] * self.support[i]
        #     support = self.support[i]
        #     # supports.append(support)
        #     if(i == 0):
        #         output = support
        #     else:
        #         output = tf.sparse_add(output,support)
        # #
        # # # output = tf.add_n(supports)
        # # output = supports
        # # output normalize
        # print type(output)
        # if(self.weight_normalize):
        #     output = normalize(output,norm='l1',axis=1)
        #
        # pre_sup = dot(x, self.vars['weights_' + str(0)],
        #               sparse=self.sparse_inputs)
        #
        # output = dot(output,pre_sup,sparse=True)
        # print type(output)
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class Wavelet_Convolution(Layer):
    """Graph convolution layer."""
    def __init__(self, node_num,weight_normalize,input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Wavelet_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num
        self.weight_normalize = weight_normalize
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))
            # diag filter kernel
            self.vars['kernel'] = ones([self.node_num], name='kernel')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]),tf.diag(self.vars['kernel']),a_is_sparse=True,b_is_sparse=True)
        supports = tf.matmul(supports,tf.sparse_tensor_to_dense(self.support[1]),a_is_sparse=True,b_is_sparse=True)
        pre_sup = dot(x, self.vars['weights_' + str(0)],sparse=self.sparse_inputs)
        output = dot(supports,pre_sup)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class Spetral_Convolution(Layer):
    """Graph convolution layer."""
    def __init__(self, node_num,weight_normalize,input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Spetral_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num
        self.weight_normalize = weight_normalize
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))
            # diag filter kernel
            self.vars['kernel'] = ones([self.node_num], name='kernel')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]),tf.diag(self.vars['kernel']),a_is_sparse=False,b_is_sparse=True)
        # supports = tf.matmul(supports,tf.sparse_tensor_to_dense(self.support[1]),a_is_sparse=False,b_is_sparse=False)
        pre_sup = dot(x, self.vars['weights_' + str(0)],
                              sparse=self.sparse_inputs)
        pre_sup = tf.matmul(tf.transpose(tf.sparse_tensor_to_dense(self.support[0])),pre_sup,a_is_sparse=False,b_is_sparse=False)
        output = dot(supports,pre_sup)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)
