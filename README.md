# Regularization-Pruning

This repository is for the new admm weight pruning methods introduced in the Neurocomputing paper:
> **A systematic DNN weight pruning framework based on symmetric accelerated stochastic ADMM [[Camera Ready](https://doi.org/10.1016/j.neucom.2024.127327)]** \
> Ming Yuan, Jianchao Bai, Feng Jiang, Lin Du\
> Northwestern Polytechnical University, Xi'an, China

TLDR: This paper introduces a new neural network pruning methods based on symmetric accelerated stochastic ADMM:
Using the advanced symmetric accelerated stochastic ADMM (SAS-ADMM), the weight pruning problem of DNNs is formulated as an optimization problem that consists of the DNN loss function and a 
regularization term. SAS-ADMM is widely used to solve the problem by dividing it into two small-dimensional and relatively easier subproblems. Besides, an optimizer based on SAS-ADMM is presented to make the DNNs after pruning converge. Experimental results demonstrate that our method achieves a faster convergence rate in a better or similar weight pruning rate than previous work. For the CIFAR-10 data set, our method reduces the number of ResNet-32 and ResNet-56 parameters by a factor of 6.61
and 9.93 while maintaining accuracy. In similar experiments of AlexNet on the ImageNet data set, we achieve 20.9x weight reduction, which only takes half of the time compared with prior work.


## Set up environment
- OS: Linux (Ubuntu 1404 and 1604 checked. It should be all right for most linux platforms. Windows and MacOS not checked.)
- python=3.6.9 (conda to manage environment is *strongly* suggested)
- All the dependant libraries are summarized in `requirements.txt`. Simply install them by `pip install -r requirements.txt`.
- CUDA (We use CUDA 10.2)

## Run the main.py file
Due to time constraints and limited resources, I will only upload the subsequent experiments of ResNet32 on CIFAR-10 gradually.

## Reference
Please cite this in your publication if our work helps your research:

    @article{yuan2024systematic,
      title={A systematic DNN weight pruning framework based on symmetric accelerated stochastic ADMM},
      author={Yuan, Ming and Bai, Jianchao and Jiang, Feng and Du, Lin},
      journal={Neurocomputing},
      volume={575},
      pages={127327},
      year={2024},
      publisher={Elsevier}
    }







