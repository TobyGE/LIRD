import pandas as pd
import numpy as np
import ast  

def read_file(filename):
    df = pd.read_csv(filename)
    state = [ast.literal_eval(i) for i in df['state'].values.tolist()]
    user = df['user'].values.tolist()
    history = [ast.literal_eval(i) for i in df['history'].values.tolist()]
    
    data = pd.DataFrame ()
    data['user'] = user
    data['state'] = state
    data['history'] = history
    return data

def old_read_file(data_path):
    ''' Load data from train.csv or test.csv. '''

    data = pd.read_csv (data_path, sep = ';')
    for col in ['user', 'state']:
        data[col] = [np.array ([[np.int(k) for k in ee.split ('&')] for ee in e.split ('|')]) for e in data[col]]
    for col in ['user', 'state']:
        data[col] = [np.array ([e[0] for e in l]) for l in data[col]]
        
    return data