"""
Description: Generating and profiling dummy models
"""

""" Import dependencies """
# Standard library imports
import copy
import os
import csv
# Third party imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # linter error
from keras.callbacks import EarlyStopping


""" 
Model functions
"""

""" Function to build model """
def build_model(neurons, n_input, num_outputs, activation_function):
    
    model = keras.Sequential()
    first_layer = neurons.pop(0)
    model.add(layers.Dense(first_layer, activation = activation_function, input_shape = [n_input]))
    
    while (len(neurons) != 0):
        layer = neurons.pop(0)
        model.add(layers.Dense(layer, activation = activation_function))
    
    model.add(layers.Dense(num_outputs))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae', 'mse'])

    return model    # note that neurons list will now be empty

"""
Function to count flops for profiling
Source: https://gist.github.com/sergeyprokudin/429c61e6536f5af5d9b0e36c660b3ae9
"""

def count_conv_params_flops(conv_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = conv_layer.output.shape.as_list()
    n_cells_total = np.prod(out_shape[1:-1])
    n_conv_params_total = conv_layer.count_params()
    conv_flops = 2 * n_conv_params_total * n_cells_total
    # if verbose:
    #     print("layer %s params: %s" % (conv_layer.name, "{:,}".format(n_conv_params_total)))
    #     print("layer %s flops: %s" % (conv_layer.name, "{:,}".format(conv_flops)))
    return n_conv_params_total, conv_flops

def count_dense_params_flops(dense_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = dense_layer.output.shape.as_list()
    _ = np.prod(out_shape[1:-1])
    n_dense_params_total = dense_layer.count_params()
    dense_flops = 2 * n_dense_params_total
    # if verbose:
    #     print("layer %s params: %s" % (dense_layer.name, "{:,}".format(n_dense_params_total)))
    #     print("layer %s flops: %s" % (dense_layer.name, "{:,}".format(dense_flops)))
    return n_dense_params_total, dense_flops

def count_model_params_flops(model):
    total_params = 0
    total_flops = 0
    model_layers = model.layers
    for layer in model_layers:
        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            params, flops = count_conv_params_flops(layer)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            params, flops = count_dense_params_flops(layer)
            total_params += params
            total_flops += flops
        else:
            print("warning:: skippring layer: %s" % str(layer))
    # print("total params (%s) : %s" % (model.name, "{:,}".format(total_params)))
    # print("total flops  (%s) : %s" % (model.name, "{:,}".format(total_flops)))
    return total_params, total_flops


'''
Write specs to CSV
'''
def write_to_file(tuple_to_write, writeFileName):
    file_path = writeFileName + ".csv"
    if not(os.path.isfile(file_path)):
        with open(file_path, 'a', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("model_name", "num_layers", "num_inputs", "num_outputs", "total_params", "total_flops"))
    with open(file_path, 'a', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(tuple_to_write)


""" Function to run instance of model training and testing """
def create_dummy_model(model_name, num_layers, num_inputs, num_outputs, window_size):
    
    activation_function = 'relu'
    if (num_inputs > num_outputs):
        neurons = [window_size*num_inputs for i in range(num_layers)]
    else:
        neurons = [window_size*num_outputs for i in range(num_layers)]

    # print("Creating dummy keras model, with parameters: model_name={}, num_layers={}, layer_config={}, num_inputs={}, num_outputs={}".format(str(model_name), str(num_layers), str(neurons), str(num_inputs), str(num_outputs)))
    
    model = build_model(neurons = neurons, n_input = num_inputs, num_outputs = num_outputs, activation_function = activation_function)
    total_params, total_flops = count_model_params_flops(model)

    # Write specs csv file
    tuple_to_write = (str(model_name), str(num_layers), str(num_inputs), str(num_outputs), str(total_params), str(total_flops))
    write_to_file(tuple_to_write = tuple_to_write, writeFileName="keras_flops")
    

    # Save model
    directory_path = os.getcwd()
    folder_path = directory_path + "\\keras_dummymodels"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model.save(folder_path + '\\' + model_name + "_" + str(num_outputs) + ".h5")

    return model