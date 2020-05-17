import itertools
import pandas as pd
import numpy as np
import random
import csv
import time


class Environment ():
    def __init__(self, data, item_embeddings, user_embeddings, alpha, gamma, fixed_length):
        self.item_embeddings = item_embeddings
		self.user_embeddings = user_embeddings

        self.embedded_data = pd.DataFrame ()
        self.embedded_data['user'] = [np.array ([embeddings.get_embedding (item_id)
                                                  for item_id in row['user']]) for _, row in data.iterrows ()]
        self.embedded_data['state'] = [np.array ([embeddings.get_embedding (item_id)
                                                   for item_id in row['state']]) for _, row in data.iterrows ()]
        self.embedded_data['n_state'] = 
        self.init_state = self.reset ()
        self.current_state = self.init_state
        self.groups = self.get_groups ()
	
	def transform(self, data):
		user = []
		state = []
		n_state = []
		for _, row in data.iterrows ():
			u = embeddings.get_embedding(row['user'][0])
			items = [embeddings.get_embedding (item_id) for item_id in row['state']]
			user.append([u])
                                                   
    def reset(self):
        init_state = self.data['state'].sample (1).values[0]
        self.current_state = init_state
        return init_state

    def step(self, actions):
        '''
        Compute reward and update state.
        Args:
          actions: embedded chosen items.
        Returns:
          cumulated_reward: overall reward.
          current_state: updated state.
        '''
		current_user = self.current_state[0]
		simulated_ratings = actions * current_user
		simulated_rewards = [1 if i >= 4 else 0 for i in simulated_ratings]
												 
        cumulated_reward = 

        # '11: Set s_t+1 = s_t' <=> self.current_state = self.current_state

        for k in range (len (simulated_rewards)):  # '12: for k = 1, K do'
            if simulated_rewards[k] > 0:  # '13: if r_t^k > 0 then'
                # '14: Add a_t^k to the end of s_t+1'
                self.current_state = np.append (self.current_state, [actions[k]], axis = 0)
                if self.fixed_length:  # '15: Remove the first item of s_t+1'
                    self.current_state = np.delete (self.current_state, 0, axis = 0)

        return simulated_rewards, cumulated_reward, self.current_state

    # Equation (4)
    def overall_reward(rewards, gamma):
        return np.sum ([gamma ** k * reward for k, reward in enumerate (rewards)])

