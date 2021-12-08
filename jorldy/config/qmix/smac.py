### DQN CartPole Config ###

env = {
    "name": "smac",
    "difficulty": "3",
}

agent = {
    "name": "qmix",
    "agent_network" : "q_network",
    "hyper_network" : "q_hyper_network",
    "mixing_network" : "q_mixing_network",
    "epsilon_init": 1.0,
    "epsilon_min": 0.05,
    "gamma": 0.99,
    "explore_ratio": 0.1,
    "buffer_size": 5000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_period": 200,
    "anneal_par": 0.000019,
    "learning_fre": 1,
    "max_grad_norm": 6,
    "learning_start_episode":33,

    "q_net_out": [64, 64],
    "mix_net_out": [32, 1],
    "q_net_hidden_size": 64,
    "shape_hyper_b2_hidden": 32,
}

optim = {
    "name": "adam",
    "lr": 0.0005,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 5000000,
    "max_episode" : 15000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    "anneal_par": 0.0004075,
    "reward_scale_par": 1,
    "per_episode_max_len": 120,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
