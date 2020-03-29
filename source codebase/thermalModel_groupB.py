"""
Version: 2.0.0
Description:
Contains script to load data from housetests datasets
    - training is fixed at 1 timesteps
"""

""" Import dependencies """
# Standard library imports
import copy
import time
# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from tqdm import tqdm_notebook
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # linter error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statistics import mean

""" 
Function for data loading (housetests)
"""
def load_preprocess_csv(filename, to_plot):
    df = pd.read_csv(filename)

    # Multiply column by 5/3600 to get Ah (this is irregardless of direction). Rename column.
    df['Charging current'] = df['Charging current'] * 5/3600
    df['Discharging current'] = df['Discharging current'] * 5/3600
    df.rename(columns={"Charging current": "Ah_CHA", "Discharging current": "Ah_DCH"}, inplace = True)

    # Check columns we have now and only keep those that we are interested in.
    df_selected = copy.deepcopy(df[['Charging', 'Seconds count', 'Ah_CHA', 'Ah_DCH', 
                                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 
                                    'Tamb', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']])

    # Get voltage averages and reorganise df_selected dataframe.
    df_selected['Voltage'] = df_selected[['V4', 'V5', 'V6', 'V7']].mean(axis=1)
    df_selected = copy.deepcopy(df_selected[['Charging', 'Seconds count', 'Ah_CHA', 'Ah_DCH', 'Voltage', 
                                            'Tamb', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']])

    # Voltages were noisy - replace out of band values with average, and take voltage to be voltage MA of 10 periods.
    df_selected['Voltage'] = df_selected['Voltage'].mask(df_selected['Voltage'] > 4.2, 3.7)
    df_selected['Voltage'] = df_selected['Voltage'].mask(df_selected['Voltage'] < 3.3, 3.7)
    df_selected[['Voltage']] = df_selected[['Voltage']].rolling(10, 1).mean()

    # Seperate charging and discharging rows and do a cummulative sum for each group while logging in-place second count for reference.
    charging_groupby = df_selected.groupby('Charging')
    for name_of_the_group, group in charging_groupby:
        if name_of_the_group == 0: # discharging
            AhAccu_DCH = group['Ah_DCH'].cumsum()
            merge_index_DCH = group['Seconds count']
        else:
            AhAccu_CHA = group['Ah_CHA'].cumsum()
            merge_index_CHA = group['Seconds count']

    # Transpose and rename columns
    temp_df_DCH = pd.DataFrame([merge_index_DCH, AhAccu_DCH]).T
    temp_df_CHA = pd.DataFrame([merge_index_CHA, AhAccu_CHA]).T
    temp_df_DCH.rename(columns={"Ah_DCH":"AhDch"}, inplace = True)
    temp_df_CHA.rename(columns={"Ah_CHA":"AhCha"}, inplace = True)

    # Merge (in order) using seconds count referehce and do a forward fill on NA values.
    df_selected = pd.merge_ordered(df_selected, temp_df_DCH, on = "Seconds count", fill_method='ffill')
    df_selected = pd.merge_ordered(df_selected, temp_df_CHA, on = "Seconds count", fill_method='ffill')

    # Merge charge and discharge currents to get a single current.
    df = pd.read_csv(filename)
    df_selected['Current'] = df['Charging current'] - df['Discharging current']

    df_selected['kWhCha'] = df_selected['AhCha'] * df_selected['Voltage']
    df_selected['kWhDch'] = df_selected['AhDch'] * df_selected['Voltage']

    # Reorder columns and fill NAs with 0s (charging in the first group would have been all NAs).
    df_selected = copy.deepcopy(df_selected[['Charging', 'Seconds count', 'Current', 'Voltage', 
                                            'AhCha', 'AhDch', 'kWhCha', 'kWhDch', 
                                            'Tamb', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']])
    df_selected.fillna(value=0, inplace = True)

    # Plot and return
    print("Data loaded!")
    if to_plot:
        df_selected[['AhCha','AhDch']].plot(figsize=(20,5), subplots = False)
        df_selected[['Current','Voltage']].plot(figsize=(20,10), subplots = True)
        df_selected[['Tamb','T1','T2','T3','T4','T5','T6','T7']].plot(figsize=(20,10), subplots = False)
    return df_selected



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
def split_sequences(sequences, n_steps, num_inputs, num_outputs):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i, 0:-num_outputs], sequences[i, num_inputs:]   # experimental: changed from -1
        X.append(seq_x)
        y.append(seq_y.flatten())
    # print(np.array(X)[0])
    # print(np.array(y)[0])
    # print(np.array(X)[1])
    # print(np.array(y)[1])
    return np.array(X), np.array(y)



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

""" Function to run instance of model training and testing """
def run_instance(model_name, num_layers, dataframe_entry, num_inputs, num_outputs, window_size, test_size, num_epochs):
    
    activation_function = 'relu'
    if (num_inputs > num_outputs):
        neurons = [window_size*num_inputs for i in range(num_layers)]
    else:
        neurons = [window_size*num_outputs for i in range(num_layers)]
    run_name = str(window_size) + '_' + str(neurons) + '_' + str(activation_function) + '_' + 'earlyStop'
    print('Run parameters: {}' .format(run_name))
    
    # Preprocessing
    X, y = split_sequences(np.array(dataframe_entry), window_size, num_inputs, num_outputs)
    # n_input = X.shape[1] * X.shape[2]
    # X = X.reshape((X.shape[0], n_input))
    X, y = seed_random(X, y)

    # print(X)
    # print(y)
    n_input = num_inputs

    if (test_size != 0):
        train_X, train_y, test_X, test_y = train_test_split(X, y, test_size)
    else:
        X = np.array(X)
        y = np.array(y)
    
    # Build and run model
    model = build_model(neurons = neurons, n_input = n_input, num_outputs = num_outputs, activation_function = activation_function)

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose = 1, restore_best_weights=True)

    start_time = time.time()
    if (test_size != 0):
        _ = model.fit(train_X, train_y, epochs = num_epochs, verbose = 0, callbacks=[es], validation_split = 0.2, shuffle = False)
    else:
        _ = model.fit(X, y, epochs = num_epochs, verbose = 0, callbacks=[es], validation_split = 0.2, shuffle = False)
    end_time = time.time()
    print("Time to train model: {} seconds".format(end_time - start_time))

    # Save and reload model
    model.save(model_name + '.h5')
    model = tf.keras.models.load_model(model_name + '.h5')

    if (test_size != 0):
        # Make predictions
        test_predictions = model.predict(test_X)
        # Save predictions
        model_pred = pd.DataFrame()
        model_pred['Actual'] = test_y
        model_pred['Predictions'] = test_predictions
        model_pred['Difference'] = model_pred['Actual'] - model_pred['Predictions']
        model_pred.reset_index(drop = True, inplace = True)
        run_stats =  model_pred.describe([0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
        return model, run_stats
    else:
        return model



def mean_error(targets, predictions):
    size_targetsArray = targets.size
    size_predictionsArray = predictions.size
    if not (size_targetsArray == size_predictionsArray):
        print("size_targetsArray does not match size_predictionsArray!")
    else:
        return (np.sum(targets - predictions))/size_targetsArray
    # return np.sqrt(((predictions - targets) ** 2).mean())

def make_estimations(model_name, dataframe_name, dataframe_entry, model, window_size, num_inputs, num_outputs):
    
    # Preprocessing
    X, y = split_sequences(np.array(dataframe_entry), window_size, num_inputs, num_outputs)
    # n_input = X.shape[1] * X.shape[2]
    # X = X.reshape((X.shape[0], n_input))

    # Make predictions
    test_predictions = model.predict(X)

    # Save predictions
    actual_temperatures_columns = ['T1a','T2a','T3a','T4a','T5a','T6a','T7a']
    predicted_temperatures_columns = ['T1p','T2p','T3p','T4p','T5p','T6p','T7p']
    actual_temperatures_df = pd.DataFrame(y, columns = actual_temperatures_columns)
    predicted_temperatures_df = pd.DataFrame(test_predictions, columns = predicted_temperatures_columns)
    merged_temperatures_df = actual_temperatures_df.merge(predicted_temperatures_df, left_index=True, right_index=True)
    merged_temperatures_df['e1'] = merged_temperatures_df['T1a'] - merged_temperatures_df['T1p']
    merged_temperatures_df['e2'] = merged_temperatures_df['T2a'] - merged_temperatures_df['T2p']
    merged_temperatures_df['e3'] = merged_temperatures_df['T3a'] - merged_temperatures_df['T3p']
    merged_temperatures_df['e4'] = merged_temperatures_df['T4a'] - merged_temperatures_df['T4p']
    merged_temperatures_df['e5'] = merged_temperatures_df['T5a'] - merged_temperatures_df['T5p']
    merged_temperatures_df['e6'] = merged_temperatures_df['T6a'] - merged_temperatures_df['T6p']
    merged_temperatures_df['e7'] = merged_temperatures_df['T7a'] - merged_temperatures_df['T7p']

    # Calculate MeanErrors
    MeanError_1 = mean_error(merged_temperatures_df['T1a'], merged_temperatures_df['T1p'])
    MeanError_2 = mean_error(merged_temperatures_df['T2a'], merged_temperatures_df['T2p'])
    MeanError_3 = mean_error(merged_temperatures_df['T3a'], merged_temperatures_df['T3p'])
    MeanError_4 = mean_error(merged_temperatures_df['T4a'], merged_temperatures_df['T4p'])
    MeanError_5 = mean_error(merged_temperatures_df['T5a'], merged_temperatures_df['T5p'])
    MeanError_6 = mean_error(merged_temperatures_df['T6a'], merged_temperatures_df['T6p'])
    MeanError_7 = mean_error(merged_temperatures_df['T7a'], merged_temperatures_df['T7p'])
    MeanError_list = [MeanError_1, MeanError_2, MeanError_3, MeanError_4, MeanError_5, MeanError_6, MeanError_7]
    print("MeanError of temperature estimations on dataset {} using model {}: \n{}, Average MeanError: {}".format(dataframe_name, model_name, str(MeanError_list), mean(MeanError_list)))
    print("\n")

    # Return dataframe of errors
    return_dataframe = merged_temperatures_df[['e1','e2','e3','e4','e5','e6','e7']]

    return return_dataframe