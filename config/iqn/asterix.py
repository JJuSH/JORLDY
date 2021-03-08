### IQN Asterix Config ###

env = {
    "name": "asterix",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "iqn",
    "network": "iqn_cnn",
    "optimizer": "adam",
    "opt_eps": 1e-2/32,
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000,

    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 20000000,
    "test_step" : 1000000,
    "print_term" : 50,
    "save_term" : 500,
}