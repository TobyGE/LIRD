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

        self.data = pd.DataFrame ()
        self.data['state'] = [np.array([user_embeddings.get_embedding(i),
                                        item_embeddings.embed(row) for i, row in data.iterrows()]
		
        self.user_history = {}
		for i, row in data.iterrows()
										= [np.array ([row for _, row in data.iterrows()]

        
        self.init_state = self.reset ()
        self.current_state = self.init_state
        self.groups = self.get_groups ()

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
		user_history = self.user_history[self.current_state[0]]
		reward = np.zeros(len(actions))
		for i in actions											 
        # '18: Compute overall reward r_t according to Equation (4)'
        simulated_rewards, cumulated_reward = self.simulate_rewards (self.current_state.reshape ((1, -1)),
                                                                     actions.reshape ((1, -1)))

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

