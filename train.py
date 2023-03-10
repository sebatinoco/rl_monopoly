from env import EconomicEnv
from agent import QLearning
from parse_args import parse_args

if __name__ == '__main__':
  
  # consolidate train arguments
  args = parse_args()
  args = vars(args)
  
  # arguments required by each class
  env_arguments = EconomicEnv.__init__.__code__.co_varnames
  agent_arguments = QLearning.__init__.__code__.co_varnames
  train_arguments = QLearning.train.__code__.co_varnames
  plot_arguments = EconomicEnv.plot.__code__.co_varnames
  
  # filter arguments
  env_args = {arg_name: arg_value for arg_name, arg_value in args.items()
              if arg_name in env_arguments}
  agent_args = {arg_name: arg_value for arg_name, arg_value in args.items()
              if arg_name in agent_arguments}
  train_args = {arg_name: arg_value for arg_name, arg_value in args.items() 
                if arg_name in train_arguments}
  plot_args = {arg_name: arg_value for arg_name, arg_value in args.items() 
              if arg_name in plot_arguments}

  # initialize environment
  env = EconomicEnv(**env_args)
  
  # initialize agent
  agent = QLearning(**agent_args, env = env)
  
  # train agent
  table = agent.train(**train_args, env = env)
  
  env.plot(**plot_args)