### Double DQN Pong MLAgent Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "double",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 450000,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 25000,
    "target_update_period": 1000,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 500000,
    "test_step" : 50000,
    "print_period" : 10,
    "save_period" : 500,
    "id": "dddqn",
    "test_iteration": 10,
}