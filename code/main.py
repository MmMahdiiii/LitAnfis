# from uci_tests.data import Haberman, Cryotheraphy
# from fq_test.classification_test import run_classification_pipeline
# import numpy as np


# test_size=0.3
# n_runs=30

# X, y = Haberman().get_data()
# model_params = {'in_features': X.shape[1],
#                 'rules': 2,
#                 'out_features': len(np.unique(y))}

# learning_params = {
#     'lr': 0.0001,
#     'batch_size' : 107,
#     'num_epochs': 500
# }

# train_performance, test_performance = run_classification_pipeline(
#     X, y,
#     test_size=test_size,
#     n_runs=n_runs,
#     model_params=model_params,
#     learning_params=learning_params,
#     min_acc=0.77,
#     model='mamdani'
# )

# print('|-----Haberman :')
# print(model_params)
# print(learning_params)
# print(f'{test_size=}, {n_runs=}')
# print('for mamadani: ')
# print('test_performance', test_performance)
# print('train_performance', train_performance)


# X, y = Cryotheraphy().get_data()
# model_params = {'in_features': X.shape[1],
#                 'rules': 2,
#                 'out_features': len(np.unique(y))}

# learning_params = {
#     'lr': 0.0005,
#     'batch_size' : 63,
#     'num_epochs': 2000
# }

# print('|-----Cryotheraphy :')
# print(model_params)
# print(learning_params)
# print(f'{test_size=}, {n_runs=}')

# train_performance, test_performance = run_classification_pipeline(
#     X, y,
#     test_size=test_size,
#     n_runs=n_runs,
#     model_params=model_params,
#     learning_params=learning_params,
#     min_acc=0.92,
#     model='mamdani'
# )
# print('for mamadani: ')
# print('test_performance', test_performance)
# print('train_performance', train_performance)

# train_performance, test_performance = run_classification_pipeline(
#     X, y,
#     test_size=test_size,
#     n_runs=n_runs,
#     model_params=model_params,
#     learning_params=learning_params,
#     min_acc=0.92,
#     model='tsk'
# )
# print('for tsk: ')
# print('test_performance', test_performance)
# print('train_performance', train_performance)


from train_test import Model_TrainTest


if __name__ == '__main__':
    # Parameters:
    train_mode = True
    render = not train_mode
    map_size = 8 # 4x4 or 8x8 
    RL_hyperparams = {
        "train_mode"            : train_mode,
        "RL_load_path"          : f'./{map_size}x{map_size}/final_weights' + '_' + '3000' + '.pth',
        "save_path"             : f'./artifacts/wights',
        "save_interval"         : 500,
        
        "clip_grad_norm"        : 3,
        "learning_rate"         : 6e-4,
        "discount_factor"       : 0.999,
        "batch_size"            : 128,
        "update_frequency"      : 10,
        "max_episodes"          : 200           if train_mode else 5,
        "max_steps"             : 200,
        "render"                : render,
        
        "epsilon_max"           : 0.99         if train_mode else -1,
        "epsilon_min"           : 0.01,
        "epsilon_decay"         : 0.999,
        
        "memory_capacity"       : 4_000        if train_mode else 0,
            
        "map_size"              : map_size,
        "num_states"            : map_size ** 2,
        "render_fps"            : 6,
        }
    
    
    # Run
    DRL = Model_TrainTest(RL_hyperparams) # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes = RL_hyperparams['max_episodes'])