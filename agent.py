import numpy as np
import random
from tqdm import tqdm

class QLearning():
  def __init__(self, env, lr, gamma):
    self.lr = lr
    self.gamma = gamma

    self.Qtable = self.initialize_q_table(env)
    
  def initialize_q_table(self, env):

    state_space = int(env.observation_space.n)
    action_space = int(env.action_space.n)

    Qtable = np.zeros((state_space, action_space))
    return Qtable

  def greedy_policy(self, Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state, :])
    
    return action

  def epsilon_greedy_policy(self, Qtable, state, epsilon, env):
    # Randomly generate a number between 0 and 1
    random_num = random.random()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
      # Take the action with the highest value given a state
      # np.argmax can be useful here
      action = self.greedy_policy(Qtable, state)
    # else --> exploration
    else:
      action = env.action_space.sample() # Take a random action
    
    return action

  def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps):
    for episode in range(n_training_episodes):
      # Reset the environment
      state = env.reset()
      step = 0
      done = False

      # repeat
      for step in tqdm(range(max_steps)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*step)

        # Choose the action At using epsilon greedy policy
        action = self.epsilon_greedy_policy(self.Qtable, state, epsilon, env)

        # Take action At and observe Rt+1 and St+1
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        self.Qtable[state, action] = self.Qtable[state, action] + self.lr * (reward + self.gamma * np.max(self.Qtable[new_state, :]) - self.Qtable[state, action])

        # If done, finish the episode
        if done:
          break
        
        # Our next state is the new state
        state = new_state
    return self.Qtable