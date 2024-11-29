from cmgp import CMGP
from cmgp.datasets import load
from cmgp.utils.metrics import sqrt_PEHE_with_diff, sqrt_PEHE
from GPy.util.multioutput import build_XY
import pandas as pd
import numpy as np

X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")

# dim = len(X_train[0])
# Dataset = pd.DataFrame(X_train)
# Dataset["Y"] = Y_train
# Dataset["T"] = W_train

# Feature_names = list(range(dim))

# Dataset0 = Dataset[Dataset["T"] == 0].copy()
# Dataset1 = Dataset[Dataset["T"] == 1].copy()

# # Extract data for the first learning task (control population)
# X0 = np.reshape(Dataset0[Feature_names].copy(), (len(Dataset0), dim))
# y0 = np.reshape(np.array(Dataset0["Y"].copy()), (len(Dataset0), 1))

# # Extract data for the second learning task (treated population)
# X1 = np.reshape(Dataset1[Feature_names].copy(), (len(Dataset1), dim))
# y1 = np.reshape(np.array(Dataset1["Y"].copy()), (len(Dataset1), 1))

# build_XY([X0, X1], [y0, y1])

model = CMGP(X_train, W_train, Y_train, max_gp_iterations=100)

pred = model.predict(X_test)

pehe = sqrt_PEHE_with_diff(Y_test, pred)

print(f"out of sample PEHE score for CMGP on ihdp = {pehe}")

pred = model.predict(X_train)

pehe = sqrt_PEHE_with_diff(Y_train_full, pred)

print(f"in sample PEHE score for CMGP on ihdp = {pehe}")
