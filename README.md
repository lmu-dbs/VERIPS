# VERIPS
This is the code repository for the paper VERIPS

# Disclaimer
This code was built by modifying [Jordan Ash's and Chicheng Zhang's Batch Active learning by Diverse Gradient Embeddings (BADGE) repository](https://github.com/JordanAsh/badge) which built upon [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning). The VGG section was supplemented with code based on [Andreas Kirsch's and Joost van Amersfoort's Batch Bald](https://github.com/BlackHC/BatchBALD) repository.

# Usage
Due to the significant amount of parameters, the VERIPS code has been restricted to running set experiments with restricted user input on the parameters.
The currently supported experiments are the ones presented in the VERIPS paper.

The code is run with:   
```python3 runVERIPS.py {TaskName} {TaskType} {TaskData} {Seed}```

{TaskName}: Experiment Setting  
- Main Results:  
  - `bald`: BALD [1] with BALD-based pseudo-labels, threshold=-0.001, decay=0
  - `badge`: BADGE [2] with BADGE-based pseudo-labels, threshold=0.0025, decay=3.3e-05
  - `entropy`: Entropy-based Active Learning [3] with Entropy-based pseudo-labels, threshold=0.05, decay=0.0033
  - `margin`: Margin-based Active Learning [3] with Margin-based pseudo-labels, threshold=0.05, decay=0
  - `rand`: random sampling, only supports Tasktype base

- Poorly Initialized Models:  
  - `entropyeq50`: Entropy-based Active Learning with Entropy-based pseudo-labels starting with 50 datapoints that are equally dsitributed across classes, threshold=0.05, decay=0.0033
  - `entropyoneout`: Entropy-based Active Learning with Entropy-based pseudo-labels starting with a starting set where one class did not receive any samples, threshold=0.05, decay=0.0033
  - `entropyimb`: Entropy-based Active Learning with Entropy-based pseudo-labels starting with a starting set where 5 of the classes did not receive any samples, threshold=0.05, decay=0.0033

- Sensitivity of Decay/Threshold:  
  - `entropy0.2`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.2, decay=0.0033
  - `entropy0.1`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.1, decay=0.0033
  - `entropy0.06`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.06, decay=0.0033
  - `entropy0.04`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.04, decay=0.0033
  - `entropy0.02`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.02, decay=0.0016
  - `entropy0.01`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.01, decay=0.0008
  - `entropydc0.0049`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.0049
  - `entropydc0.004`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.004
  - `entropydc0.002`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.002
  - `entropydc0.001`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0.001
  - `entropydc0`: Entropy-based Active Learning with Entropy-based pseudo-labels, threshold=0.05, decay=0
 
{TaskType}: Experiment Type
- `base`: Basic Active Learning without any pseudo-labels
- `ceal`: Reimplementation of CEAL [4]
- `verips`: VERIPS
- `fverips`: Accelerated version of VERIPS
  
{TaskData}: Dataset for the Experiment
- `CIFAR10`
- `SVHN`
- `MNIST`
- `CIFAR10s`: Modified CIFAR10 with 10% of samples for 50% of the classes (randomly chosen)

{Seed}: seed of the Experiment (in Paper: 0 to 4)

Each experiment produces a log file which tracks its parameters in the file name and is put into a logs folder (that is created automatically if it does not exist).  
Please also note that there are a lot of outputs that refer to specific metrics as well as some remaining debug outputs. 

# Requirements

Make sure you install all requirements using:
```
pip install -U pip setuptools wheel
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

# References
[1] Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. "Deep bayesian active learning with image data." (2017).  
[2] Ash, Jordan T., et al. "Deep batch active learning by diverse, uncertain gradient lower bounds." (2019).   
[3] Settles, Burr. "Active learning literature survey." (2009).  
[4] Wang, Keze, et al. "Cost-effective active learning for deep image classification." (2016).  

# Citing
If you use our code in your research or applications, please consider citing our paper.:
```
@inproceedings{gilhuber2022verips,
  author       = {Sandra Gilhuber and
                  Philipp Jahn and
                  Yunpu Ma and
                  Thomas Seidl},
  title        = {{VERIPS:} Verified Pseudo-label Selection for Deep Active Learning},
  booktitle    = {{ICDM}},
  pages        = {951--956},
  publisher    = {{IEEE}},
  year         = {2022}
}
```
