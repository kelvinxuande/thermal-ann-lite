"""
Version: 1.1.0
Description:
Contains script to load data from LDPRF datasets
"""

""" Import dependencies """
# Standard library imports
import copy
from datetime import datetime
# Third party imports
import numpy as np
import pandas as pd

""" 
Function for data loading (LDPRF)
"""
def load_csv(filename, features_list, mode):
    df = pd.read_csv(filename, header = 12, dtype = str)
    # df = df.loc[:, data_list]
    df.dropna(axis=0, how='any', inplace = True)

    # minute, second seperation:
    df['Program time'] = df['Program time'].str.split(':')
    df[['minute','second']] = pd.DataFrame(df['Program time'].values.tolist(), index = df.index)
    df['new_set'] = pd.eval('df.minute.str.contains("00") and df.second.str.contains("00.0")', engine='python')
    df.drop(columns = ['Program time'], inplace = True)

    if (mode == 0):
        search_list = ['AhCha','AhDch','Temp','minute','second']
    elif (mode == 1):
        search_list = ['Current','Voltage','Temp','minute','second']
    elif (mode == 2):
        search_list = ['Current','Voltage','AhCha','AhDch','Temp','minute','second']
    df[search_list] = df[search_list].apply(pd.to_numeric, errors='raise')
    df[search_list] = df[search_list].astype('float64', copy = True, errors = 'raise')

    df.reset_index(drop = True, inplace = True)
    # get index of new sets
    # this indexes to the next needs to be added with 30 secs (see above)
    set_index = df.index[df['new_set']].tolist()
    del set_index[0:3]
    second_increment = [round(df['second'][i-1] - df['second'][i-2], 2) for i in set_index]
    # index of new sets and everything after, increase by second_increment
    for index in range(len(second_increment)):
        if (index != len(second_increment) - 1):
            df['second'][set_index[index]:set_index[index+1]] = df['second'][set_index[index]:set_index[index+1]] + second_increment[index]
        else:
            df['second'][set_index[index]:] = df['second'][set_index[index]:] + second_increment[index]

    # now, we form a list of new sets - 1
    prev_index = [i -1 for i in set_index]
    # take values of indexes new set - 1, add to indexes next set and everything after
    seconds_summation = [(df['minute'][i] * 60) + df['second'][i] for i in prev_index]
    for index in range(len(seconds_summation)):
        df['second'][set_index[index]:] = df['second'][set_index[index]:] + seconds_summation[index]
    # finally, convert all minutes to seconds
    df['second'] = df['second'] + df['minute'] * 60

    # do some clean-ups
    df.drop(columns = ['minute', 'new_set'], inplace = True)
    df['Amb'] = df['Temp'].min()

    if (mode == 0):
        df = df[['second','AhCha','AhDch','Amb','Temp']]
    elif (mode == 1):
        df = df[['second','Current','Voltage','Amb','Temp']]
    elif (mode == 2):
        df = df[['second','Current','Voltage','AhCha','AhDch','Amb','Temp']]

    df.columns = features_list
    return df


""" Function to plot predictions """
def extract_complexity(nested_model_dictionary, nested_errors_dictionary):
    NN_size = []
    mean_error = []

    for key, value in nested_model_dictionary.items():
        for key_1, value_1 in value.items():
            NN_size.append(value_1.count_params())
            mean_error.append(nested_errors_dictionary[key][key_1])

    df_2d = []
    for percentage_reduction in range(len(mean_error)):
        # repack:
        df_1d = [(percentage_reduction + 1) * 10, NN_size[percentage_reduction], mean_error[percentage_reduction]]
        df_2d.append(df_1d)
    df = pd.DataFrame(df_2d, columns = ['Percentage_reduced', 'NN_size','mean_error'])
    
    return df


# """ Function to plot predictions """
# def extract_complexity(nested_model_dictionary, nested_errors_dictionary, percentile):
#     sizes = []
#     errors = []

#     try:
#         lower_percentile = str(100 - percentile) + '%'
#     except Exception as e:
#         print(e)
#         lower_percentile = 'min'
#     percentile = str(percentile) + '%'

#     for key in nested_model_dictionary.keys():
#         sizes_1 = []
#         errors_1 = []

#         for key_1, value in nested_model_dictionary[key].items():
#             size_error = value.count_params()
#             sizes_1.append(size_error)

#             upperBoundError = nested_errors_dictionary[key][key_1].at[percentile]
#             lowerBoundError = nested_errors_dictionary[key][key_1].at[lower_percentile]
#             error_size = max(abs(upperBoundError), abs(lowerBoundError))
#             errors_1.append(error_size)

#         sizes.append(sizes_1)
#         errors.append(errors_1)

#     df_2d = []
#     for precentage_reduction in range(len(errors)):
#         for timesteps in range(len(errors[0])):
#             NN_size = sizes[precentage_reduction][timesteps]
#             AE = errors[precentage_reduction][timesteps]
#             df_1d = [(precentage_reduction + 1) * 10, timesteps + 1, NN_size, AE]
#             df_2d.append(df_1d)
#     df = pd.DataFrame(df_2d, columns = ['Percentage_reduced', 'Timesteps', 'NN_size','Absolute_error'])
    
#     return df