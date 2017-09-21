import gym
from gym import wrappers
import tensorflow as tf
import os
import pickle
import sys

from net import Net
from pcl import PCL
from replay_buffer import ReplayBuffer

training_type = sys.argv[1]

if training_type == "online":
    load_from_file = 0
elif training_type == "offline":
    load_from_file = 1
elif training_type == "online_offline":
    load_from_file = 0

off_policy_rate = 20
log_dir = './log'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
env = gym.make('Acrobot-v1')
env_spec = dict(
        action_space=env.action_space,
        observation_space=env.observation_space)

#env = wrappers.Monitor(env, log_dir, force=True)

net = Net(env_spec)

replay_buffer = ReplayBuffer(load_from_file=load_from_file)

sess = tf.Session()

agent = PCL(1000, env, env_spec, replay_buffer, sess, net, load_from_file=load_from_file, off_policy_rate=off_policy_rate, mode=training_type)

rewards = agent.train()

pickle.dump(rewards, open("rewards_" + training_type + ".p", "wb"))


if training_type == "online_offline":
    pickle.dump(replay_buffer.buffer, open("buffer.p", "wb"))
    pickle.dump(replay_buffer.weight, open("weight.p", "wb"))
