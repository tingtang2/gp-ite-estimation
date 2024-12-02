from cmgp import CMGP
from cmgp.datasets import load
from cmgp.utils.metrics import sqrt_PEHE_with_diff, sqrt_PEHE
from cmgp.utils.random import enable_reproducible_results

X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")

# repeat across multiple random seeds
for i in range(3):
    enable_reproducible_results(i)
    model = CMGP(X_train, W_train, Y_train, max_gp_iterations=100)

    pred = model.predict(X_test)

    pehe = sqrt_PEHE_with_diff(Y_test, pred)

    print(f"out of sample PEHE score for CMGP on ihdp = {pehe}")

    pred = model.predict(X_train)

    pehe = sqrt_PEHE_with_diff(Y_train_full, pred)

    print(f"in sample PEHE score for CMGP on ihdp = {pehe}")

# repeat across multiple random seeds
for i in range(3):
    enable_reproducible_results(i)
    model = CMGP(X_train,
                 W_train,
                 Y_train,
                 sparse_mode='sgpr',
                 max_gp_iterations=100)

    pred = model.predict(X_test)

    pehe = sqrt_PEHE_with_diff(Y_test, pred)

    print(f"out of sample PEHE score for CMGP on ihdp = {pehe}")

    pred = model.predict(X_train)

    pehe = sqrt_PEHE_with_diff(Y_train_full, pred)

    print(f"in sample PEHE score for CMGP on ihdp = {pehe}")
