import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MaternKernel, LCMKernel
import torch
from gpytorch import likelihoods

from gpytorch.models.computation_aware_gp import ComputationAwareGP


class CaGP(ComputationAwareGP):

    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor,
                 likelihood, projection_dim: int, input_dim: int,
                 num_tasks: int):
        mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        # covar_module = LCMKernel([
        #     MaternKernel(nu=1.5, ard_num_dims=input_dim),
        #     MaternKernel(nu=1.5, ard_num_dims=input_dim)
        # ],
        #  num_tasks=num_tasks)
        covar_module = MaternKernel(nu=1.5, ard_num_dims=input_dim)

        super(CaGP, self).__init__(train_inputs=train_inputs,
                                   train_targets=train_targets,
                                   mean_module=mean_module,
                                   covar_module=covar_module,
                                   likelihood=likelihood,
                                   projection_dim=projection_dim)
