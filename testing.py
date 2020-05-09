import itertools
import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout

def state_to_items(state, actor, ra_length, embeddings, dict_embeddings, target=False):
    return [dict_embeddings[str(action)]
            for action in actor.get_recommendation_list(ra_length, np.array(state).reshape(1, -1), embeddings, target).reshape(ra_length, embeddings.size())]


def test_actor(actor, test_df, embeddings, dict_embeddings, ra_length, history_length, target=False, nb_rounds=1):
    ratings = []
    unknown = 0
    random_seen = []
    for _ in range (nb_rounds):
        for i in range (len (test_df)):
            history_sample = list (test_df[i].sample (history_length)['itemId'])
            recommendation = state_to_items (embeddings.embed (history_sample), actor, ra_length, embeddings,
                                             dict_embeddings, target)
            for item in recommendation:
                l = list (test_df[i].loc[test_df[i]['itemId'] == item]['rating'])
                assert (len (l) < 2)
                if len (l) == 0:
                    unknown += 1
                else:
                    ratings.append (l[0])
            for item in history_sample:
                random_seen.append (list (test_df[i].loc[test_df[i]['itemId'] == item]['rating'])[0])

    return ratings, unknown, random_seen
