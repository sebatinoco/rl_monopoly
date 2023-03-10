import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class EconomicEnv(gym.Env):
  metadata = {'render_modes': None}

  def __init__(self, A: float, c: float, F = 0, elasticity = 2, n_actions = 10, n_convergence = 10, render_mode = None):
    self.A = A # disposicion total a pagar
    self.c = c # costo marginal
    self.F = F # costo fijo
    self.elasticity = elasticity # elasticidad demanda

    self.observation_space = spaces.Discrete(int(A - F)) # todas las cantidades posibles
    self.action_space = spaces.Discrete(n_actions) # todos los precios posibles

    self.n_convergence = n_convergence # N veces el mismo precio para detener experimento

    self._action_to_price = {action: int(action * (A - F)/(n_actions - 1) + F) for action in range(n_actions)} # A to price
    self._q_to_state = {self.demand(self._action_to_price[action]): action for action in range(len(self._action_to_price))} # Q to state

    self.metric = [] # lista con la métrica a evaluar
    self.prices = [] # lista con los precios elegidos
    self.quantities = [] # lista con las cantidades generadas

    self.price_m = int(minimize(self.get_root, 0, method = 'SLSQP').x)

  def demand(self, P):
    '''
    Función que recibe un precio, devuelve la cantidad vendida según la función de demanda
    '''

    Q = self.A - P * self.elasticity if (self.A - P * self.elasticity) > 0 else 0

    return Q

  def get_root(self, P):

    Q = self.A - P * self.elasticity

    fun = Q * P - self.cost(Q)

    return -fun

  def cost(self, Q):
    '''
    Función que recibe una cantidad, devuelve el costo total de producir dicha cantidad.
    '''

    C = self.c * Q + self.F

    return C

  def get_revenue(self):

    r = self._p * self._q - self.cost(self._q)

    return r

  def total_revenue(self):
    return sum(self.metric)

  def _get_obs(self):
    '''
    Función que devuelve el estado representando la cantidad vendida.
    '''

    return self._q_to_state[self._q]

  def _get_info(self):
    return self.total_revenue()

  def reset(self, seed = None):
    super().reset(seed = seed)

    self.convergence_count = 0

    random_action = random.randint(0, self.action_space.n - 1)

    self._p = self._action_to_price[random_action]
    self.prices.append(self._p)

    self._q = self.demand(self._p)

    observation = self._get_obs()
    
    return observation

  def step(self, action):
    # acción a precio
    self._p = self._action_to_price[action]

    # terminar loop si precio se repite mas de convergence_count veces
    if self._p == self.prices[-1]:
      self.convergence_count += 1
    else:
      self.convergence_count = 0
    terminated = True if self.convergence_count >= self.n_convergence else False

    self.prices.append(self._p) # concatenamos precio elegido

    self._q = self.demand(self._p) # la cantidad que se vende
    self.quantities.append(self._q) # concatenamos cantidad obtenida

    observation = self._get_obs() # transformacion a estado

    reward = self.get_revenue()
    self.metric.append(reward)

    info = self._get_info()

    return observation, reward, terminated, info

  def plot(self, exp_name: str, plots_dir: str, window_size = 200):
    rolling_mean = np.convolve(self.prices, np.ones(window_size)/window_size, mode='valid')

    fill = np.full([window_size - 1], np.nan)
    rolling_mean = np.concatenate((fill, rolling_mean))

    plt.figure(figsize = (15, 6))
    #plt.plot(range(len(env.prices)), env.prices, label = 'Actual Price')
    plt.plot(range(len(self.prices)), rolling_mean, label = 'Moving Average')
    plt.axhline(y = self.price_m, color = 'r', linestyle = '--', label = 'Monopoly Price')
    plt.axhline(y = self.c, color = 'g', linestyle = '--', label = 'Marginal Cost')
    plt.ylabel('Price')
    plt.xlabel('Step')
    plt.title('Solving Microeconomic Environment with Q-learning')
    plt.legend()
    
    plt.savefig(f'{plots_dir}/{exp_name}.png', transparent = False, bbox_inches = 'tight')