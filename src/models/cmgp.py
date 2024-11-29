import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import RBFKernel, LCMKernel


class CMGPModel(ExactGP):

    def __init__(self, train_x, train_y, likelihood, num_tasks, input_dim):
        super(CMGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = LCMKernel([
            RBFKernel(ard_num_dims=input_dim),
            RBFKernel(ard_num_dims=input_dim)
        ],
                                      num_tasks=num_tasks)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(
            mean_x, covar_x)
