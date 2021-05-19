# Error-Modulated STDP

## Getting Started

The project contains three files:
1. EMSTDP_algo:
    - contains the main algorithm
    - Network class encapsulates the the whole network, its parameters and hyperparameters
        - provides initialization, train and test functions
    - an SNN network is implemented as a Network object

2. EMSTDP_main:
    - builds, trains and tests a SNN network with EMSTDP
    - helpful mainly for debugging

3. EMSTDP_disGridSearch:
    - used for distributed grid search
    - utilizes Ray Tuner which is a Python library for hyperparameter tuning at any scale
        - https://ray.readthedocs.io/en/latest/tune.html
        - allows for visualization through Tensorboard, checkpoints and various scalable hyperparameter search algorithms
    - can be used single runs as well

### Prerequisites

The code has been tested on python 2.7.15 with the following requirements

```
numpy==1.15.1
ray==0.5.3
Keras==2.2.4
tqdm==4.26.0
matplotlib==2.2.3
opt_einsum==0+untagged.25.geee2bb2
tensorboard_logger==0.1.0

```

## Training and Testing steps

 - modify the algorithm or the training procedure in EMSTDP_algo.py
 - debug with a specific set of hyperparameters using EMSTDP_main.py. Commented code are available to visualize the
   weights and spiking activity of the feed forward and the error network
 - provide a grid of hyperparameters, number of cpu cores per instance, search algorithm and the relative/absolute path
   in EMSTDP_disGridSearch.py to run the grid search, visualize through tensorboard and checkpoints

