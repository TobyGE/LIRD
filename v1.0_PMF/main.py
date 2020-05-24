# from data_generator import DataGenerator
# from embeddings_generator import EmbeddingsGenerator
from data_util import read_file
from embeddings import Embeddings
import tensorflow as tf
from training_process import *
from environment import Environment
from actor import *
from critic import *
from replay import ReplayMemory
import os


# Hyperparameters

# if __name__ == '__main__':

history_length = 5 # N in article
ra_length = 4 # K in article
discount_factor = 0.99 # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001 # τ in Algorithm 3
batch_size = 64
nb_episodes = 50
nb_rounds = 100
filename_summary = 'summary.txt'
alpha = 0.5 # α (alpha) in Equation (1)
gamma = 0.9 # Γ (Gamma) in Equation (4)
buffer_size = 1000000 # Size of replay memory D in article
fixed_length = True # Fixed memory length

# dg = DataGenerator('ml-100k/u.data', 'ml-100k/u.item')
# dg.gen_train_test(0.8, seed=42)

# dg.write_csv('train.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
# dg.write_csv('test.csv', dg.test, nb_states=[history_length], nb_actions=[ra_length])

# data = read_file(os.path.dirname(os.getcwd())+'/data/ml-100k/train.csv')

data = read_file('./data/ml-100k/train_data.csv')

# if True: # Generate embeddings
#   eg = EmbeddingsGenerator(dg.user_train, pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']))
#   eg.train(nb_epochs=300)
#   train_loss, train_accuracy = eg.test(dg.user_train)
#   print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
#   test_loss, test_accuracy = eg.test(dg.user_test)
#   print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
#   eg.save_embeddings('embeddings.csv')



item_embeddings = Embeddings(np.load('./data/ml-100k/item_embed.npy'))
user_embeddings = Embeddings(np.load('./data/ml-100k/user_embed.npy'))

print('Successfully load training data and embeddings!')

state_space_size = item_embeddings.size() * history_length
action_space_size = item_embeddings.size() * ra_length

env_args = {}
env_args['data'] = data
env_args['item_embeddings'] = item_embeddings
env_args['user_embeddings'] = user_embeddings
env_args['gamma'] = gamma
environment = Environment(**env_args)

tf.compat.v1.reset_default_graph() # For multiple consecutive executions

if tf.test.is_gpu_available():
	config = tf.ConfigProto(device_count = {'GPU': 0})
	sess = tf.Session(config=config)
else:
	sess = tf.Session()
# '1: Initialize actor network f_θ^π and critic network Q(s, a|θ^µ) with random weights'
actor = Actor(sess, state_space_size, action_space_size, batch_size, ra_length, history_length, item_embeddings.size(), tau, actor_lr)
critic = Critic(sess, state_space_size, action_space_size, history_length, item_embeddings.size(), tau, critic_lr)

train(sess, environment, actor, critic, item_embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary, nb_rounds, **env_args)



# dict_embeddings = {}
# for i, item in enumerate (embeddings.get_embedding_vector ()):
#     str_item = str (item)
#     assert (str_item not in dict_embeddings)
#     dict_embeddings[str_item] = i
