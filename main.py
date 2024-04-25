from uci_tests.iris.iris import Iris    
from fq_test.classification_test import run_classification_pipeline
import numpy as np


X, y = Iris().get_data()
model_params = {'in_features': X.shape[1],
                'rules': 2,
                'out_features': len(np.unique(y))}

learning_params = {
    'lr': 0.001,
    'batch_size' : 4,
    'num_epochs': 300
}

train_performance, test_performance = run_classification_pipeline(
    X, y,
    test_size=0.3,
    n_runs=30,
    model_params=model_params,
    learning_params=learning_params
)

print(test_performance)
print(train_performance)