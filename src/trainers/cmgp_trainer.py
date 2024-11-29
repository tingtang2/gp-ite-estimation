import torch
from models.cmgp import CMGPModel
from models.ca_gp import CaGP
import gpytorch
from gpytorch.mlls import ComputationAwareELBO, ComputationAwareIterativeELBO
import pandas as pd
import numpy as np
from GPy.util.multioutput import build_XY
import logging
from tqdm import trange

from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood


class CMGPTrainer:

    def __init__(self,
                 X,
                 Treatments,
                 Y,
                 max_gp_iterations=1000,
                 device=torch.device('cuda'),
                 model_type='ca_gp'):
        """
        Initialize a GPyTorch model for causal inference.

        Args:
            X (np.ndarray): Input covariates.
            Treatments (np.ndarray): Treatment assignments.
            Y (np.ndarray): Observed outcomes.
            max_gp_iterations (int): Maximum number of optimization iterations.
        """
        X = torch.tensor(X)
        Treatments = torch.tensor(Treatments, dtype=torch.long)
        Y = torch.tensor(Y)

        self.dim = X.shape[1] + 1
        # self.num_tasks = len(torch.unique(Treatments))
        self.num_tasks = 1
        self.max_gp_iterations = max_gp_iterations
        self.device = device
        self.model_type = model_type

        self._fit(X, Treatments, Y)

    def _preprocess(self, X_train, Y_train, W_train):
        '''
        prepreprocessing function from CMGP repo
        '''

        dim = len(X_train[0])
        Dataset = pd.DataFrame(X_train)
        Dataset["Y"] = Y_train
        Dataset["T"] = W_train

        Feature_names = list(range(dim))

        Dataset0 = Dataset[Dataset["T"] == 0].copy()
        Dataset1 = Dataset[Dataset["T"] == 1].copy()

        # Extract data for the first learning task (control population)
        X0 = np.reshape(Dataset0[Feature_names].copy(), (len(Dataset0), dim))
        y0 = np.reshape(np.array(Dataset0["Y"].copy()), (len(Dataset0), 1))

        # Extract data for the second learning task (treated population)
        X1 = np.reshape(Dataset1[Feature_names].copy(), (len(Dataset1), dim))
        y1 = np.reshape(np.array(Dataset1["Y"].copy()), (len(Dataset1), 1))

        X, Y, _ = build_XY([X0, X1], [y0, y1])
        return torch.from_numpy(X).to(self.device), torch.from_numpy(Y).to(
            self.device)

    def _fit(self, Train_X, Train_T, Train_Y):
        """
        Fit the CMGP model using the training data.
        """
        # Combine treatment and input data for multitask structure
        train_x, train_y = self._preprocess(Train_X, Train_Y, Train_T)

        # Initialize the model and likelihood
        # self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
        self.likelihood = GaussianLikelihood()

        if self.model_type == 'ca_gp':
            self.model = CaGP(train_x, train_y, self.likelihood, 1, self.dim,
                              self.num_tasks).to(self.device)
        else:
            self.model = CMGPModel(train_x, train_y, self.likelihood,
                                   self.num_tasks, self.dim).to(self.device)

        # Use the Adam optimizer
        self.optimizer = torch.optim.Adam(
            [{
                "params": self.model.parameters()
            }],
            lr=0.1,
        )

        # "Loss" for GPs is the marginal log likelihood
        if self.model_type == 'ca_gp':
            self.mll = ComputationAwareELBO(self.likelihood, self.model)
            # self.mll = ComputationAwareIterativeELBO(self.likelihood,
            #  self.model)
        else:
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.model)

        # Training loop
        self.model.train()
        self.likelihood.train()

        for i in trange(self.max_gp_iterations):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            logging.info('Iter %d/%d - Loss: %.3f' %
                         (i + 1, self.max_gp_iterations, loss.item()))
            self.optimizer.step()

    def predict(self, X):
        """
        Predict the treatment effect for input covariates X.
        """
        self.model.eval()
        self.likelihood.eval()
        X = torch.from_numpy(X).to(self.device)

        with torch.no_grad():
            task_0_x = torch.cat(
                [X, torch.zeros(X.size(0), 1).to(self.device)], dim=1)
            task_1_x = torch.cat(
                [X, torch.ones(X.size(0), 1).to(self.device)], dim=1)

            mean_0 = self.model(task_0_x).mean
            mean_1 = self.model(task_1_x).mean

            treatment_effect = mean_1 - mean_0
        return treatment_effect.detach().cpu().numpy()
