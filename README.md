# gp-ite-estimation
Individualized treatment effects estimation with Gaussian Processes

## Setup

Please install libraries in the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```

Use of virtual environments is optional but recommended. For GPU support refer to [the PyTorch docs to ensure correct the CUDA version](https://pytorch.org/get-started/locally/).

Also be sure to specify/change the `save_dir` and `data_dir` configs in the `main.py`.

## Relevant papers:
- [Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes](https://arxiv.org/abs/1704.02801)
    - [Code Link](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/causal_multitask_gaussian_processes_ite)
- [Computation-Aware Gaussian Processes: Model Selection And Linear-Time Inference](https://arxiv.org/abs/2411.01036)
