# VERIPS
This is the code repository for the paper VERIPS

# Disclaimer
This code was built by modifying [Jordan Ash's and Chicheng Zhang's Batch Active learning by Diverse Gradient Embeddings (BADGE) repository](https://github.com/JordanAsh/badge) which built upon [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning).

# Usage
Due to the significant amount of parameters, the VERIPS code has been restricted to running set experiments with restricted user input on the parameters.
The currently supported experiments are the ones presented in the VERIPS paper.

The code is run with 
python3 runVERIPS.py TaskName TaskType TaskData Seed

TaskName: Experiment Setting
  bald: BALD with BALD-based pseudo-labels, threshold=-0.001, decay=0
  badge: BADGE with BADGE-based pseudo-labels, threshold=0.0025, decay=3.3e-05
  entropy: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.0033
  margin: Margin-based Active Learning with Margin-based pseudo-labels, threshold=0.05, decay=0
  entropyeq50: Entropy-based Active Learning with Entropy-based pseudo-labels starting with 50 datapoints that are equally dsitributed across classes, threshold=0.05, decay=0.0033
  entropyoneout: Entropy-based Active Learning with Entropy-based pseudo-labels starting with a starting set where one class did not receive any samples, threshold=0.05, decay=0.0033
  entropyimb: Entropy-based Active Learning with Entropy-based pseudo-labels starting with a starting set where 5 of the classes did not receive any samples, threshold=0.05, decay=0.0033
  entropy0.1: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.1, decay=0.0033
  entropy0.06: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.06, decay=0.0033
  entropy0.04: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.04, decay=0.0033
  entropydc0.004: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.004
  entropydc0.002: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.002
  entropydc0: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0
 
 TaskType: Experiment Type
  base: basic Active Learning without any pseudo-labels
  ceal: reimplementation of CEAL
  verips: VERIPS
  fverips: FastVERIPS
  
 TaskData: Dataset for the Experiment
  CIFAR10
  SVHN
  MNIST

Please also note that there are a lot of outputs that refer to specific metrics as well as some remaining debug outputs.