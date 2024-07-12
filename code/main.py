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


from imitation import Imitaition
# import utilities.data_downloader

from utilities.config import imitaition_conf

im = Imitaition(imitaition_conf)
im.train()

