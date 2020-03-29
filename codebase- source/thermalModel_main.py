"""
Version: 4.3.0
Description:
Contains commonly used functions in project
"""

""" Import dependencies """
# Standard library imports
import copy
from datetime import datetime
import time
# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # linter error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



""" 
Functions for data preprocessing
"""

""" Splitting data into test and training sets """
def train_test_split(X, y, test_size):
    
    train_X = np.array(X[:-test_size])
    train_y = np.array(y[:-test_size])
    test_X = np.array(X[-test_size:])
    test_y = np.array(y[-test_size:])
    
    # print(train_X.shape, train_y.shape)
    # print(test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y

def train_test_split_norm(X, y, test_size, train_mean, train_var):
    
    train_X = np.array(X[:-test_size])
    train_y = np.array(y[:-test_size])
    test_X = np.array(X[-test_size:])
    test_y = np.array(y[-test_size:])
    
    # print(train_X.shape, train_y.shape)
    # print(test_X.shape, test_y.shape)

    # data normalisation (use to it to only calculate mean and variance with training dataset)
    if len(train_mean) == 0:
        train_mean = stats.describe(train_X).mean
        train_var = stats.describe(train_X).variance
    train_X = (train_X - train_mean) / train_var
    test_X = (test_X - train_mean) / train_var

    return train_X, train_y, test_X, test_y, train_mean, train_var

""" Shuffle and map input variables list to target variables list """
def seed_random(a, b):
    c = list(zip(a, b))
    np.random.seed(0)
    np.random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)

""" Example input:
x = [[1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7]]
y = [1, 2, 3, 4, 5]
X, y = seed_random(x, y)
Expected output:
[[5, 6, 7], [4, 5, 6], [2, 3, 4], [1, 2, 3], [3, 4, 5]]
[5, 4, 2, 1, 3]
"""

""" Spitting long/ entire sequence into shorter subsequences """
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



""" 
Model functions
"""

""" Function to build model """
def build_model(neurons, n_input, activation_function):
    
    model = keras.Sequential()
    first_layer = neurons.pop(0)
    model.add(layers.Dense(first_layer, activation = activation_function, input_shape = [n_input]))
    
    while (len(neurons) != 0):
        layer = neurons.pop(0)
        model.add(layers.Dense(layer, activation = activation_function))
    
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae', 'mse'])

    return model    # note that neurons list will now be empty


def mean_error(targets, predictions):
    size_targetsArray = targets.size
    size_predictionsArray = predictions.size
    if not (size_targetsArray == size_predictionsArray):
        print("size_targetsArray does not match size_predictionsArray!")
    else:
        return (np.sum(targets - predictions))/size_targetsArray
    # return np.sqrt(((predictions - targets) ** 2).mean())


""" Function to run instance of model training and testing """
def run_instance(model_name, num_layers, dataframe_entry, num_inputs, window_size, test_size, num_epochs):
    
    activation_function = 'relu'
    neurons = [window_size*num_inputs for i in range(num_layers)]
    run_name = str(window_size) + '_' + str(neurons) + '_' + str(activation_function) + '_' + 'earlyStop'
    print('Run parameters: {}' .format(run_name))
    
    # Preprocessing
    X, y = split_sequences(np.array(dataframe_entry), window_size)
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    X, y = seed_random(X, y)

    # if (test_size != 0):
    #     train_X, train_y, test_X, test_y = train_test_split(X, y, test_size)
    # else:
    X = np.array(X)
    y = np.array(y)
    
    # Build and run model
    model = build_model(neurons = neurons, n_input = n_input, activation_function = activation_function)

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose = 1, restore_best_weights=True)
    # if (test_size != 0):
    #     _ = model.fit(train_X, train_y, epochs = num_epochs, verbose = 0, callbacks=[es], validation_split = 0.2, shuffle = False)
    # else:

    start_time = time.time()
    _ = model.fit(X, y, epochs = num_epochs, verbose = 0, callbacks=[es], validation_split = 0.2, shuffle = False)
    end_time = time.time()
    print("Time to train model: {} seconds".format(end_time - start_time))

    # Save and reload model
    model.save(model_name + '.h5')
    model = tf.keras.models.load_model(model_name + '.h5')

    return model



""" Run instances in a predefined loop """
def loop_run_instances(identifier, loop_name, num_layers, train_dataframe, test_dataframe, num_inputs, start_window_size, end_window_size, window_size_step, test_size, num_epochs):
    model_dictionary = {}
    model_mean_error_dict = {}

    for window_size in range(start_window_size, end_window_size + window_size_step, window_size_step):
        model_name = identifier + '_' + loop_name + '_' + str(window_size)
        model = run_instance(model_name = model_name, num_layers = num_layers, dataframe_entry = train_dataframe, num_inputs = num_inputs, window_size = window_size, test_size = test_size, num_epochs = num_epochs)
        model_mean_error = predictions_plot(model_name = model_name, dataframe_entry = test_dataframe, model = model, window_size = window_size, plot_errors = False)

        model_dictionary[model_name] = model
        model_mean_error_dict[model_name] = model_mean_error
    
    return model_dictionary, model_mean_error_dict



""" Function to plot predictions """
def predictions_plot(model_name, dataframe_entry, model, window_size, plot_errors):
    
    # Preprocessing
    X, y = split_sequences(np.array(dataframe_entry), window_size)
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))

    # Make predictions
    test_predictions = model.predict(X)
    # print(test_predictions.shape)

    # Save predictions
    model_df_temp = pd.DataFrame()
    model_df_temp['Actual'] = y
    model_df_temp['Predictions'] = test_predictions
    model_mean_error = mean_error(model_df_temp['Actual'], model_df_temp['Predictions'])

    if (plot_errors):
        model_df_temp['Difference'] = model_df_temp['Actual'] - model_df_temp['Predictions']
        model_df_temp.reset_index(drop = True, inplace = True)
        run_stats =  model_df_temp['Difference'].describe([0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
        model_df_temp_plot = copy.deepcopy(model_df_temp)
        exclude = ['Difference']
        model_df_temp_plot.loc[:, model_df_temp_plot.columns.difference(exclude)].plot(title=model_name, subplots=False, figsize=(20,10))

    return model_mean_error



""" Function to plot predictions """
def noise_testing(model_name, dataframe_entry, model, window_size, plot_errors):
    
    # Preprocessing
    X, y = split_sequences(np.array(dataframe_entry), window_size)
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))

    # Generate noised inputs
    mu_1, sigma_1 = 0, 0.50
    noise_1 = pd.Series(np.random.normal(mu_1, sigma_1, size = X.shape[0]*X.shape[1]))
    noise_1 = noise_1.values.reshape((X.shape[0], X.shape[1]))
    X_1 = X + noise_1

    mu_2, sigma_2 = 0, 0.33
    noise_2 = pd.Series(np.random.normal(mu_2, sigma_2, size = X.shape[0]*X.shape[1]))
    noise_2 = noise_2.values.reshape((X.shape[0], X.shape[1]))
    X_2 = X + noise_2

    mu_3, sigma_3 = 0, 1.66
    noise_3 = pd.Series(np.random.normal(mu_3, sigma_3, size = X.shape[0]*X.shape[1]))
    noise_3 = noise_3.values.reshape((X.shape[0], X.shape[1]))
    X_3 = X + noise_3

    # Make predictions
    test_predictions = model.predict(X)
    test_predictions_1 = model.predict(X_1)
    test_predictions_2 = model.predict(X_2)
    test_predictions_3 = model.predict(X_3)

    # Save predictions
    model_df_temp = pd.DataFrame()
    model_df_temp['Actual'] = y
    model_df_temp['Predictions'] = test_predictions
    model_df_temp['Predictions_1'] = test_predictions_1
    model_df_temp['Predictions_2'] = test_predictions_2
    model_df_temp['Predictions_3'] = test_predictions_3
    model_mean_error = mean_error(model_df_temp['Actual'], model_df_temp['Predictions'])
    model_mean_error_1 = mean_error(model_df_temp['Actual'], model_df_temp['Predictions_1'])
    model_mean_error_2 = mean_error(model_df_temp['Actual'], model_df_temp['Predictions_2'])
    model_mean_error_3 = mean_error(model_df_temp['Actual'], model_df_temp['Predictions_3'])

    return [model_mean_error, model_mean_error_1, model_mean_error_2, model_mean_error_3]


def get_error_dataframe(mean_error_dictionary):
    list_2d = []
    for key, value in mean_error_dictionary.items():
        list_1d = []
        list_1d.append(key)
        list_1d.extend(value)
        list_2d.append(list_1d)

    df = pd.DataFrame(list_2d, columns = ['Model_name', 'mean_error_clean', 'mean_error_1', 'mean_error_2' ,'mean_error_3'])
    
    return df



# """ Function to plot predictions """
# def plot_complexity_errors(plot_name, model_dictionary, errors_dictionary, percentile):
#     sizes = []
#     errors = []

#     try:
#         lower_percentile = str(100 - percentile) + '%'
#     except:
#         lower_percentile = 'min'
#     percentile = str(percentile) + '%'

#     for key, value in model_dictionary.items():
#         size_error = value.count_params()
#         sizes.append(size_error)
#         upperBoundError = errors_dictionary[key].at[percentile]
#         lowerBoundError = errors_dictionary[key].at[lower_percentile]
#         error_size = max(abs(upperBoundError), abs(lowerBoundError))
#         errors.append(error_size)
    
#     df = pd.DataFrame(list(zip(sizes, errors)), columns =['Neural Network size', 'Absolute Error'])

#     # x 'independent cause', y 'dependent effect'
#     df.plot(x = 'Neural Network size', y = 'Absolute Error', title = plot_name, subplots=False, figsize=(20,10))

#     return df

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
    if verbose:
        print("layer %s params: %s" % (conv_layer.name, "{:,}".format(n_conv_params_total)))
        print("layer %s flops: %s" % (conv_layer.name, "{:,}".format(conv_flops)))
    return n_conv_params_total, conv_flops

def count_dense_params_flops(dense_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = dense_layer.output.shape.as_list()
    _ = np.prod(out_shape[1:-1])
    n_dense_params_total = dense_layer.count_params()
    dense_flops = 2 * n_dense_params_total
    if verbose:
        print("layer %s params: %s" % (dense_layer.name, "{:,}".format(n_dense_params_total)))
        print("layer %s flops: %s" % (dense_layer.name, "{:,}".format(dense_flops)))
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
    print("total params (%s) : %s" % (model.name, "{:,}".format(total_params)))
    print("total flops  (%s) : %s" % (model.name, "{:,}".format(total_flops)))
    return total_params, total_flops