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
        self.data['state'] = [np.array ([user_embeddings.get_embedding(row[0]),
                                        item_embeddings.embed(row[1:]) for _, row in data.iterrows ()]
        self.data['user_history'] = [np.array ([row for _, row in data.iterrows ()]

        
        self.init_state = self.reset ()
        self.current_state = self.init_state
        self.groups = self.get_groups ()

    def reset(self):
        init_state = self.embedded_data['state'].sample (1).values[0]
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

    def simulate_rewards(self, current_state, chosen_actions, reward_type='grouped cosine'):
        '''
        Calculate simulated rewards.
        Args:
          current_state: history, list of embedded items.
          chosen_actions: embedded chosen items.
          reward_type: from ['normal', 'grouped average', 'grouped cosine'].
        Returns:
          returned_rewards: most probable rewards.
          cumulated_reward: probability weighted rewards.
        '''

        # Equation (1)
        def cosine_state_action(s_t, a_t, s_i, a_i):
            cosine_state = np.dot (s_t, s_i.T) / (np.linalg.norm (s_t, 2) * np.linalg.norm (s_i, 2))
            cosine_action = np.dot (a_t, a_i.T) / (np.linalg.norm (a_t, 2) * np.linalg.norm (a_i, 2))
            return (self.alpha * cosine_state + (1 - self.alpha) * cosine_action).reshape ((1,))

        if reward_type == 'normal':
            # Calculate simulated reward in normal way: Equation (2)
            probabilities = [cosine_state_action (current_state, chosen_actions, row['state'], row['action'])
                             for _, row in self.embedded_data.iterrows ()]
        elif reward_type == 'grouped average':
            # Calculate simulated reward by grouped average: Equation (3)
            probabilities = np.array ([g['size'] for g in self.groups]) * \
                            [(self.alpha * (
                                        np.dot (current_state, g['average state'].T) / np.linalg.norm (current_state, 2)) \
                              + (1 - self.alpha) * (
                                          np.dot (chosen_actions, g['average action'].T) / np.linalg.norm (chosen_actions,
                                                                                                           2)))
                             for g in self.groups]
        elif reward_type == 'grouped cosine':
            # Calculate simulated reward by grouped cosine: Equations (1) and (3)
            probabilities = [cosine_state_action (current_state, chosen_actions, g['average state'], g['average action'])
                             for g in self.groups]

        # Normalize (sum to 1)
        probabilities = np.array (probabilities) / sum (probabilities)

        # Get most probable rewards
        if reward_type == 'normal':
            returned_rewards = self.embedded_data.iloc[np.argmax (probabilities)]['reward']
        elif reward_type in ['grouped average', 'grouped cosine']:
            returned_rewards = self.groups[np.argmax (probabilities)]['rewards']

        

        if reward_type in ['normal', 'grouped average']:
            # Get cumulated reward: Equation (4)
            cumulated_reward = overall_reward (returned_rewards, self.gamma)
        elif reward_type == 'grouped cosine':
            # Get probability weighted cumulated reward
#             cumulated_reward = np.sum ([p * overall_reward (g['rewards'], self.gamma)
#                                         for p, g in zip (probabilities, self.groups)])
            cumulated_reward = overall_reward (returned_rewards, self.gamma)

        return returned_rewards, cumulated_reward
    
    
