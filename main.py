from uci_tests.iris.iris import Iris    
from fq_test.classification_test import run_classification_pipeline
import numpy as np


X, y = Iris().get_data()
model_params = {'in_features': X.shape[1],
                'rules': 2,
                'out_features': len(np.unique(y))}

learning_params = {
    'lr': 0.007,
    'batch_size' : 16,
    'num_epochs': 100
}

train_performance, test_performance = run_classification_pipeline(
    X, y,
    n_splits=10,
    n_runs=3,
    model_params=model_params,
    learning_params=learning_params
)

print(np.mean(test_performance))
print(np.mean(train_performance))