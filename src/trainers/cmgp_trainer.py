import torch
from models.cmgp import CMGPModel
import gpytorch
from cmgp.datasets import load

from gpytorch.likelihoods import MultitaskGaussianLikelihood


class CMGPTrainer:

    def __init__(self, X, Treatments, Y, max_gp_iterations=1000):
        """
        Initialize a GPyTorch model for causal inference.

        Args:
            X (np.ndarray): Input covariates.
            Treatments (np.ndarray): Treatment assignments.
            Y (np.ndarray): Observed outcomes.
            max_gp_iterations (int): Maximum number of optimization iterations.
        """
        X = torch.tensor(X, dtype=torch.float)
        Treatments = torch.tensor(Treatments, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.float)

        self.dim = X.shape[1]
        self.num_tasks = len(torch.unique(Treatments))
        self.max_gp_iterations = max_gp_iterations

        self._fit(X, Treatments, Y)

    def _fit(self, Train_X, Train_T, Train_Y):
        """
        Fit the CMGP model using the training data.
        """
        # Combine treatment and input data for multitask structure
        task_indices = Train_T.view(-1, 1)  # Column vector of task indices
        train_x = torch.cat([Train_X, task_indices], dim=1)
        train_y = Train_Y

        # Initialize the model and likelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
        self.model = CMGPModel(train_x, train_y, self.likelihood,
                               self.num_tasks, self.dim)

        # Use the Adam optimizer
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters()
                },
                {
                    "params": self.likelihood.parameters()
                },
            ],
            lr=0.1,
        )

        # "Loss" for GPs is the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)

        # Training loop
        self.model.train()
        self.likelihood.train()

        for i in range(self.max_gp_iterations):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        """
        Predict the treatment effect for input covariates X.
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            task_0_x = torch.cat([X, torch.zeros(X.size(0), 1)], dim=1)
            task_1_x = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)

            mean_0 = self.model(task_0_x).mean
            mean_1 = self.model(task_1_x).mean

            treatment_effect = mean_1 - mean_0
        return treatment_effect.numpy()
