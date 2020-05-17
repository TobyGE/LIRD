import itertools
import pandas as pd
import numpy as np
import random
import csv
import time
# from data_util import read_file
# import ast  

class Environment ():
	def __init__(self, data, item_embeddings, user_embeddings, gamma):
		self.data = data
		self.item_embeddings = item_embeddings
		self.user_embeddings = user_embeddings
		self.gamma = gamma

		self.embedded_data = pd.DataFrame()
		self.embedded_data['user'] = [np.array(user_embeddings.get_embedding(row['user'])) for _, row in data.iterrows ()]
		self.embedded_data['state'] = [np.array(item_embeddings.embed(row['state'])) for _, row in data.iterrows ()]
		self.embedded_data['history'] = [np.array(item_embeddings.embed(row['history'])) for _, row in data.iterrows ()]

		self.init_state = self.reset(0)

		self.item_embed_dict = {}
		for i, embed in enumerate(item_embeddings.get_embedding_vector()):
			self.item_embed_dict[tuple(embed)] = i


	def reset(self, idx):
		init_user, _, init_history = self.data.loc[idx].values.tolist()
		_, init_state, _ = self.embedded_data.loc[idx].values.tolist()
		self.current_user = init_user
		self.current_state = init_state
		self.current_history = init_history
		return self.current_state
	#         return init_user, init_state

	def step(self, actions, item_idxes):
		'''
		Compute reward and update state.
		Args:
		  actions: embedded chosen items. actions.shape = (nb_actions, item_embed_size)
		Returns:
		  cumulated_reward: overall reward.
		  current_state: updated state.
		'''
	# 		current_user_embed = self.user_embeddings.get_embedding(self.current_user)
		
# 		actions_to_items = [self.item_embed_dict[tuple(i)] for i in actions]
# 		print(actions_to_items,item_idxes)
# 		input()
		
		historic_rewards = [1 if idx in self.current_history else 0 for idx in item_idxes]
		cumulated_reward = self.overall_reward(historic_rewards, self.gamma)

	# 		simulated_ratings = actions * current_user_embed
	# 		simulated_rewards = [1 if i >= 4 else 0 for i in simulated_ratings]


		# '11: Set s_t+1 = s_t' <=> self.current_state = self.current_state
		for k in range (len(historic_rewards)):  # '12: for k = 1, K do'
			if historic_rewards[k] > 0:  # '13: if r_t^k > 0 then'
				# '14: Add a_t^k to the end of s_t+1'
				self.current_state = np.append (self.current_state, [actions[k]], axis = 0)
# 				if self.fixed_length:  # '15: Remove the first item of s_t+1'
				self.current_state = np.delete (self.current_state, 0, axis = 0)

		return historic_rewards, cumulated_reward, self.current_state

	# Equation (4)
	def overall_reward(self, rewards, gamma):
		return np.sum ([gamma ** k * reward for k, reward in enumerate (rewards)])

