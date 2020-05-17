import itertools
import pandas as pd
import numpy as np
import random
import csv
import time
import copy
import pandas as pd

import matplotlib.pyplot as plt


class DataPreprocessor():
	def __init__(self, datapath, itempath):
		'''
		Load data from the DB MovieLens
		List the users and the items
		List all the users historic
		'''
		self.data  = self.load_datas(datapath, itempath)
		self.users = self.data['userId'].unique()   #list of all users
		self.items = self.data['itemId'].unique()   #list of all items

		#a list contains the rating history of each user
		self.histo = self.gen_histo()


	def load_datas(self, datapath, itempath):
		'''
		Load the data and merge the name of each movie.
		A row corresponds to a rate given by a user to a movie.

		 Parameters
		----------
		datapath :  string
					path to the data 100k MovieLens
					contains usersId;itemId;rating
		itempath :  string
					path to the data 100k MovieLens
					contains itemId;itemName
		 Returns
		-------
		result :    DataFrame
					Contains all the ratings
		'''
		data = pd.read_csv(datapath, sep='\t',
					   names=['userId', 'itemId', 'rating', 'timestamp'])
		movie_titles = pd.read_csv(itempath, sep='|', names=['itemId', 'itemName'],
						   usecols=range(2), encoding='latin-1')
		return data.merge(movie_titles,on='itemId', how='left')


	def gen_histo(self):
		'''
		Group all rates given by users and store them from older to most recent.

		Returns
		-------
		result :    List(DataFrame)
					List of the historic for each user
		'''
		historic_users = []
		for i, u in enumerate (self.users):
			temp = self.data[self.data['userId'] == u]
			temp = temp.sort_values ('timestamp').reset_index ()
			temp.drop ('index', axis = 1, inplace = True)
			historic_users.append (temp)
		return historic_users

	
	def sample_histo_v5(self, user_histo, nb_states, pivot_rating=4):
		prop_histo = user_histo[user_histo['rating'] >= pivot_rating]
		if len(prop_histo) > nb_states:
			user = user_histo['userId'][0] - 1
			initial_state =  prop_histo[0:nb_states]['itemId'].values.tolist()
			user_history =  prop_histo[nb_states:]['itemId'].values.tolist()
		return user, initial_state, user_history
	


	def write_csv(self, train_test_ratio=0.8, nb_states=5):
		users = []
		initial_states = []
		user_histories = []
		
		print(len(self.histo))
		
		for user_histo in self.histo:
			try:
				user, init_state, u_history = self.sample_histo_v5(user_histo, nb_states)
				users.append(user)
				initial_states.append(init_state)
				user_histories.append(u_history)
			except:
				continue
				
		train_data = pd.DataFrame()
		test_data = pd.DataFrame()
		
		train_data['user'] = users
		train_data['state'] = initial_states
		train_data['history'] = [u_h[0:int(train_test_ratio*len(u_h))] for u_h in user_histories]

		test_data['user'] = users
		test_data['state'] = initial_states
		test_data['history'] = [u_h[int(train_test_ratio*len(u_h)):] for u_h in user_histories]
		
		
		train_data.to_csv('./data/ml-100k/train_data.csv', index=False)
		test_data.to_csv('./data/ml-100k/test_data.csv', index=False)
		


