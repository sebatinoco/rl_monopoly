import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    
    # env arguments
    parser.add_argument('--A', type = float, default = 100.0, help = 'max disposition to pay of demmand')
    parser.add_argument('--c', type = float, default = 1.0, help = 'marginal cost of agents')
    parser.add_argument('--elasticity', type = float, default = 1.0, help = 'elasticity of the demmand')
    parser.add_argument('--n_actions', type = int, default = 10, 
                        help = 'number of actions (prices) available to the agent')
    parser.add_argument('--n_convergence', type = int, default = 100,
                        help = 'min steps to conclude convergence')
    
    # agent arguments
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate of the agent')
    parser.add_argument('--gamma', type = float, default = 0.95, help = 'discount factor of the agent')
    
    # training arguments
    parser.add_argument('--n_training_episodes', type = int, default = 1, 
                        help = 'number of training episodes')
    parser.add_argument('--min_epsilon', type = float, default = 0.05, 
                        help = 'min epsilon of the experiment')
    parser.add_argument('--max_epsilon', type = float, default = 1.0,
                        help = 'max epsilon of the experiment')
    parser.add_argument('--decay_rate', type = float, default = 5e-4,
                        help = 'decay rate of the exploration phase')
    parser.add_argument('--max_steps', type = int, default = int(2.5e5), 
                        help = 'max steps of the experiment')
    
    # plot arguments
    parser.add_argument("--exp_name", type = str, default = 'monopoly',
    help="the name of this experiment")
    parser.add_argument('--plots_dir', type = str, default = 'plots', help = 'folder location to save plot results')
    parser.add_argument('--window_size', type = int, default = 200, help = 'size of window to plot moving average')
    
    args = parser.parse_args()
    
    return args