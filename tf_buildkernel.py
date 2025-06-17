import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
buildkernel_module = tf.load_op_library(os.path.join(base_dir, 'tf_buildkernel_so.so'))


def spherical_kernel(database,
                     query,
                     nn_index,
                     nn_count,
                     nn_dist,
                     radius,
                     kernel=[8,2,3]):
    '''
    Input:
        database: (batch, npoint, 3+) float32 array, database points (x,y,z,...)
        query:    (batch, mpoint, 3+) float32 array, query points (x,y,z,...)
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        nn_dist: (batch, mpoint, nnsample) float32, sqrt distance array
        radius:  float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (batch, mpoint, nnsample) int32 array, filter bin indices
    '''
    n, p, q = kernel

    database = database[:, :, 0:3]  #(x,y,z)
    query = query[:, :, 0:3] #(x,y,z)
    return buildkernel_module.spherical_kernel(database, query, nn_index, nn_count, nn_dist, radius, n, p, q)
ops.NoGradient('SphericalKernel')


def fuzzy_spherical_kernel(database,
                           query,
                           nn_index,
                           nn_count,
                           nn_dist,
                           radius,
                           kernel=[8,2,3]):
    '''
    Input:
        database: (batch, npoint, 3+) float32 array, database points (x,y,z,...)
        query:    (batch, mpoint, 3+) float32 array, query points (x,y,z,...)
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        nn_dist: (batch, mpoint, nnsample) float32, sqrt distance array
        radius:  float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (batch, mpoint, nnsample, 8) int32 array, fuzzy filter indices,
                    (8=2*2*2, 2 for each splitting dimension)
        filt_coeff: (batch, mpoint, nnsample, 8) float32 array, fuzzy filter weights,
                    kernelsize=prod(kernel)+1
    '''
    n, p, q = kernel

    database = database[:, :, 0:3]  #(x,y,z)
    query = query[:, :, 0:3] #(x,y,z)
    return buildkernel_module.fuzzy_spherical_kernel(database, query, nn_index, nn_count, nn_dist, radius, n, p, q)
ops.NoGradient('FuzzySphericalKernel')


def kpconv_kernel(database,
                  query,
                  kernel_points,
                  nn_index,
                  nn_count,
                  radius):
    sigma = radius/2.5
    kernel_points = kernel_points*(1.5*sigma)
    database = database[:, :, 0:3]  #(x,y,z)
    query = query[:, :, 0:3] #(x,y,z)
    return buildkernel_module.kpconv_kernel(database, query, kernel_points, nn_index, nn_count, sigma)
ops.NoGradient('KpconvKernel')


def fuzzy_kpconv_kernel(database,
                        query,
                        kernel_points,
                        nn_index,
                        nn_count,
                        radius):
    sigma = radius/2.5
    kernel_points = kernel_points*(1.5*sigma)
    database = database[:, :, 0:3]  #(x,y,z)
    query = query[:, :, 0:3] #(x,y,z)
    return buildkernel_module.fuzzy_kpconv_kernel(database, query, kernel_points, nn_index, nn_count, sigma)
ops.NoGradient('FuzzyKpconvKernel')


import tensorflow as tf
from tensorflow.python.framework import ops
import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
conv3d_module = tf.load_op_library(os.path.join(base_dir, 'tf_conv3d_so.so'))

def depthwise_conv3d(input, filter, nn_index, nn_count, bin_index):
    '''
    Input:
        input:   (batch, npoint, in_channels) float32 array, input point features
        filter: (binsize, in_channels, channel_multiplier) float32 array, convolution filter
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        bin_index: (batch, mpoint, nnsample), filtet bins' indices
    Output:
        output: (batch, mpoint, out_channels) float32 array, output point features
    '''
    return conv3d_module.depthwise_conv3d(input, filter, nn_index, nn_count, bin_index)

@ops.RegisterGradient("DepthwiseConv3d")
def _depthwise_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    nn_index = op.inputs[2]
    nn_count = op.inputs[3]
    bin_index = op.inputs[4]
    grad_input, grad_filter = conv3d_module.depthwise_conv3d_grad(input, filter, grad_output, nn_index,
                                                                  nn_count, bin_index)
    return [grad_input, grad_filter, None, None, None]


def fuzzy_depthwise_conv3d(input, filter, nn_index, nn_count, bin_index, bin_coeff):
    '''
    Input:
        input:   (batch, npoint, in_channels) float32 array, input point features
        filter: (binsize, in_channels, channel_multiplier) float32 array, convolution filter
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        bin_index: (batch, mpoint, nnsample, 8) int32 array, filtet bins' indices
        bin_coeff: (batch, mpoint, nnsample, 8) float32 array, kernel bin coefficients
    Output:
        output: (batch, mpoint, out_channels) float32 array, output point features
    '''
    return conv3d_module.fuzzy_depthwise_conv3d(input, filter, nn_index, nn_count,
                                                bin_index, bin_coeff)

@ops.RegisterGradient("FuzzyDepthwiseConv3d")
def _fuzzy_depthwise_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    nn_index = op.inputs[2]
    nn_count = op.inputs[3]
    bin_index = op.inputs[4]
    bin_coeff = op.inputs[5]
    grad_input, grad_filter = conv3d_module.fuzzy_depthwise_conv3d_grad(input, filter, grad_output,
                                                                        nn_index, nn_count,
                                                                        bin_index, bin_coeff)
    return [grad_input, grad_filter, None, None, None, None]