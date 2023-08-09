# Learning Specialized Activation Functions for Physics-informed Neural Networks

This repository contains the PyTorch source code for the experiments in the manuscript:

[Learning Specialized Activation Functions for Physics-informed Neural Networks](https://arxiv.org/abs/2308.04073).

## Introduction

In this work, we reveal the connection between the optimization difficulty of PINNs and activation functions. Specifically, we show that PINNs exhibit high sensitivity to activation functions when solving PDEs with distinct properties. Existing works usually choose activation functions by inefficient trial-and-error. To avoid the inefficient manual selection and to alleviate the optimization difficulty of PINNs, we introduce adaptive activation functions to search for the optimal function when solving different problems. We compare different adaptive activation functions and discuss their limitations in the context of PINNs. Furthermore, we propose to tailor the idea of learning combinations of candidate activation functions to the PINNs optimization, which has a higher requirement for the smoothness and diversity on learned functions. This is achieved by removing activation functions which cannot provide higher-order derivatives from the candidate set and incorporating elementary functions with different properties according to our prior knowledge about the PDE at hand. We further enhance the search space with adaptive slopes. The proposed adaptive activation function can be used to solve different PDE systems in an interpretable way. Its effectiveness is demonstrated on a series of benchmarks.

## Installation

```
git clone git@github.com:LeapLabTHU/AdaAFforPINNs.git
cd AdaAFforPINNs
conda create -n myenv python=3.9
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate myenv
conda install --file=requirements.txt
```

## Instructions

Training on the convection, Allen-Cahn, KdV, or Cahn-Hilliard equations.
```
cd source
bash scripts/run.sh 
```

<!-- 
## Citation
If you find our project useful in your research, please consider citing:

```text
@article{krishnapriyan2021characterizing,
  title={Characterizing possible failure modes in physics-informed neural networks},
  author={Krishnapriyan, Aditi S. and Gholami, Amir and Zhe, Shandian and Kirby, Robert and Mahoney, Michael W},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
``` -->

## Contact 
If you have any question, please feel free to contact the authors. Honghui Wang: wanghh20@mails.tsinghua.edu.cn.

## Acknowledgments
This codebase is built on the repository of [Characterizing possible failure modes in physics-informed neural networks](https://github.com/a1k12/characterizing-pinns-failure-modes). Please consider citing or starring the project.
