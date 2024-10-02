import plotly.express as px
import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import base64
import segyio
import time
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import itertools



def find_files_in_directory(dir_path, ext):
    """
    Find all files in a directory with a specific file extension.

    @param dir_path: path to the directory
    @param ext: file extension to search for
    @return: list of files with the specified extension in the directory
    """
    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith(ext):
            res.append(file)
    return res


def std_mean_normalization(input_data):
    """_summary_

    Args:
        input_data (array): _description_

    Returns:
        array: _description_
    """
    return (input_data - np.mean(input_data)) / np.std(input_data)


def get_mask(os, t_dim=[128,128,128]):
    """
    Create a mask of specified dimensions with values decaying from 1 to 0 at the edges.

    @param os: length of the edge transition
    @param t_dim: dimensions of the mask (default: [128,128,128])
    @return: 3D mask of specified dimensions
    """
    # training image dimensions
    n1, n2, n3 = t_dim[0], t_dim[1], t_dim[2]
    # initialize mask with all 1's
    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1

    # create decay values for edge transition
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)

    # apply decay values to edges of mask
    for k1 in range(os):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k1]
                sc[n1-k1-1][k2][k3]=sp[k1]
    for k1 in range(n1):
        for k2 in range(os):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k2]
                sc[k1][n3-k2-1][k3]=sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(os):
                sc[k1][k2][k3]=sp[k3]
                sc[k1][k2][n3-k3-1]=sp[k3]
    return sc



def predict_with_mask(loaded_model, input_data, os=12, normalize_patch=False, t_dim=[128,128,128]):
    """ function that predicts the whole slice with a window-based approach 
        code adapted from https://github.com/xinwucwp/faultSeg

    Args:
        loaded_model (tensorflow): neural network
        input_data (array): the whole slice
        os (int, optional): overlap width. Defaults to 12.
        normalize_patch (bool, optional): if to do MinMax normalization on each patch. Defaults to False.
        t_dim (list, optional): training image dimensions. Defaults to [128,128,128].

    Returns:
        _type_: _description_
    """
    # training image dimensions
    n1, n2, n3 = t_dim[0], t_dim[1], t_dim[2]

    # input_data dimensions
    m1,m2, m3 = input_data.shape[0], input_data.shape[1], input_data.shape[2]
    
    c1 = int(np.round((m1+os)/(n1-os)+0.5))
    c2 = int(np.round((m2+os)/(n2-os)+0.5))
    c3 = int(np.round((m3+os)/(n3-os)+0.5))

    p1 = (n1-os)*c1+os
    p2 = (n2-os)*c2+os
    p3 = (n3-os)*c3+os

    input_data = np.reshape(input_data,(m1,m2,m3))
    gp = np.zeros((p1,p2,p3),dtype=np.single)
    predict = np.zeros((p1,p2,p3),dtype=np.single)
    mk = np.zeros((p1,p2,p3),dtype=np.single)
    gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
    gp[0:m1,0:m2,0:m3] = input_data

    sc = get_mask(os, t_dim)

    my_bar = st.progress(0)
    metric = st.empty()
    counter = 0
    t_start = 0
    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                counter += 1
                with metric.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Patches out of {}".format(c1*c2*c3), value=counter)
                    col2.metric(label="Estimate wait time", value= "--" if t_start == 0 else time.strftime('%H:%M:%S',time.gmtime(((t_end - t_start)*(c1*c2*c3-counter)))))
                    col3.metric(label="Seconds to predict 1 patch", value= "--" if t_start == 0 else str(round((t_end - t_start), 2)))
                my_bar.progress(counter/(c1*c2*c3))
                t_start = time.time()

                b1 = k1*n1-k1*os
                e1 = b1+n1
                b2 = k2*n2-k2*os
                e2 = b2+n2
                b3 = k3*n3-k3*os
                e3 = b3+n3
                gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
                if normalize_patch:
                    gs = gs-np.min(gs)
                    gs = gs/np.max(gs)
                    gs = gs*255 
                Y = loaded_model.predict(gs,verbose=1)
                Y = np.array(Y)
                predict[b1:e1,b2:e2,b3:e3]= predict[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
                mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
                t_end = time.time()
    predict = predict/mk
    predict = predict[0:m1,0:m2,0:m3]
    return input_data, predict


def save_to_numpy(file_path, numpy_data):
    np.save(file_path, numpy_data)

def save_to_segy_3d(original_segy, file_path, numpy_data, session_state):
    #TODO make sure that input seismic is segy
    input_sepredict = original_segy.get_file_name()
    output_sepredict = file_path+".sgy"

    with segyio.open(input_sepredict, \
        iline=original_segy.get_iline_byte(), xline=original_segy.get_xline_byte()) as src:
        spec = segyio.spec()

        spec.sorting = int(src.sorting)

        spec.format = int(src.format)
        spec.samples =  src.samples[:]
        spec.tracecount = src.tracecount

        spec.ilines = src.ilines
        spec.xlines = src.xlines

        cropped_info = session_state['cropped_info']
        ny, nz = original_segy.get_n_xlines(), original_segy.get_n_zslices()

        with segyio.create(output_sepredict, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(src.ext_headers):
                dst.text[i] = src.text[i]
            # Copy the binary header, then insert the modifications needed for the new time axis
            dst.bin = src.bin
            # Copy all trace headers to destination file
            dst.header = src.header 
            # Copy all trace headers to destination file
            dst.header.iline = src.header.iline

            if cropped_info is None:
                for itrace in range(dst.tracecount):
                    dst.header[itrace] =  src.header[itrace]
                    dst.trace[itrace] = np.zeros(len(src.samples)).astype('float32') 
                iter = 0
                for i in range(src.ilines.min(), src.ilines.max()):
                    data = numpy_data[iter]
                    dst.iline[i] = data.astype('float32')
                    iter = iter + 1
            else:
                for itrace in range(dst.tracecount):
                    dst.header[itrace] =  src.header[itrace]
                    dst.trace[itrace] = np.zeros(len(src.samples)).astype('float32')
                iter = 0
                for i in range(src.ilines.min() + cropped_info[0,0], src.ilines.min() + cropped_info[0,1] ): # +1
                    data = np.zeros([ny, nz]) 
                    data[cropped_info[1,0]:cropped_info[1,1], cropped_info[2,0]:cropped_info[2,1]] = numpy_data[iter]
                    dst.iline[i] = data.astype('float32')
                    iter = iter + 1


                
def _to_tensor(x, dtype):
   
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def cross_entropy_balanced(y_true, y_pred):
   
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)