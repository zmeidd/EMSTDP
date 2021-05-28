CONTENTS OF THIS EMSTDP
---------------------
This repo consists of both python and Loihi implementations of EMSTDP. The author of EMSTDP is Amar Shrestha.

**Python Implementaion of EMSTDP**
-------
1. EMSTDP_algo: 
   - contains the main algorithm 
   - Network class encapsulates the the whole network, its parameters and hyperparameters
      -  provides initialization, train and test functions
   - An SNN network is implemented as a Network object 
   
2. EMSTDP_main:
    - Builds, trains and tests a SNN network with EMSTDP
    - Helpful mainly for debugging
    - Contains a EMSTDP fully connected nerual network with layer size 784-100-10, which tested MNIST dataset
3. EMSTDP_disGridSearch:
    - used for distributed grid search
    - utilizes Ray Tuner which is a Python library for hyperparameter tuning at any scale
        - https://ray.readthedocs.io/en/latest/tune.html
        - allows for visualization through Tensorboard, checkpoints and various scalable hyperparameter search algorithms
    - can be used single runs as well

---
**Run**
---
```
python3 EMSTDP_main.py 
```
**Dependencies**
- numpy==1.15.1
- ray==0.5.3
- Keras==2.2.4
- tqdm==4.26.0
- matplotlib==2.2.3
- opt_einsum==0+untagged.25.geee2bb2
- tensorboard_logger==0.1.0

**Loihi Implementaion of EMSTDP**
-----
- expel_net.py
  - Builds the layerwise neuron group connections for EMSTDP
   -  setupNetwork(self, train, conv_wgt, wgt, bwgt)
      - train: boolean argument, True for training
      - conv_wgt: pretrained convolutional weights
      - wgt: Initialized weights, used for feedforward path
      - bwgt: backpropagation weights
      
    - genTrainingData(): Generating training data, the built in datasets are MNIST, MSTAR 7 classes and MSTAR 10 classes
    
- epl_parameters.py
   - Set up the parameters for EMSTDP loihi learning
      -numInputs: Input neuron number
      -numHidNurns Numbers of Neurons in each hidden layers
      -numTargets: Number of the output neurons
      
 - epl_snips_utils.py 
   - Snips file used for training , testing and generating input, output bias
   - Changing snips settings in this file
   
 - epl_multi_pattern_learning.py 
   - Builds an entire EMSTDP network on Loihi
   
 - neural_network.py:
   -2 layers Neural network applied to MNIST
   - MSTAR, MNIST data preprocessing functions.
-----












