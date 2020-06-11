# Code for **Stochastic Security: Adversarial Defense Using Long-Run Dynamics of Energy-Based Models**

This repository contains PyTorch implementations of the training and evaluation experiments from our paper:

**Stochastic Security: Adversarial Defense Using Long-Run Dynamics of Energy-Based Models**<br/>Mitch Hill\*, Jonathan Mitchell\*, and Song-Chun Zhu (*\*equal contributions*)<br/>https://arxiv.org/abs/2005.13525.

The file ```bpda_eot_attack.py``` is an implementation of the BPDA+EOT attack to evaluate EBM defense. The files ```train_ebm.py``` and ```train_clf.py``` will train an EBM and classifier network respectively.

**NOTE:** All config files use the pixel range [0, 255] for adversarial perturbation ```adv_eps``` and attack step size ```adv_eta```. However, all experiments scale images so that the pixels range is  [-1, 1]. Adversarial parameters are scaled accordingly during execution. The Langevin step size ```langevin_eps``` in the config files uses the pixel range [-1, 1].

## Attack Code

The BPDA+EOT attack in Algorithm 1 of our paper is implemented by ```bpda_eot_attack.py```. The folder ```config_attack``` has three files for three different attacks. The config ```bpda_eot_attack.json``` is the main attack used against EBM defene in our paper (see Section 4.2). The config ```clf_nat_attack.json``` is set up evaluate the base classifier without purification. The config ```clf_transfer_attack.json``` will create adversarial samples from the base classifier before a final evaluation using EBM purification (see Section 4.1). 

## EBM Training Code

The EBM training procedure in Algorithm 2 of our paper is implemented by ```train_ebm.py```. The implementation is heavily based on [this repository](https://github.com/point0bar1/ebm-anatomy) . The main difference is the introduction of an initial phase of non-convergent learning with the Adam optimizer followed by a final phase of convergent learning using SGD (see Section 2). The folder ```config_train_ebm``` has training files for Cifar-10, SVHN, and Cifar-100.

## Classifier Training Code

The file ```train_clf.py``` is a minimal implementation of natural and adversarial training for classifier networks. The folder ```config_train_clf``` has config files for natural and adversarial training for Cifar-10, SVHN, and Cifar-100. Setting the config parameter ```adv_steps_train``` to 0 will lead to natural training, while a positive value will lead to adversarial training.

## Running an Experiment

To run an experiment with ```bpda_eot_attack.py```, ```train_ebm.py```, or ```train_clf.py```, just specify the JSON config file:

```python
# json file with experiment config
CONFIG_FILE = './path_to/config.json'
```

before execution.

## Other Files

Network structures are located in ```nets.py```. Experiment setup a dataset import function are in ```utils.py```.

## Pre-trained Networks

A pre-trained natural classifier ```clf.pth``` and EBM ```ebm.pth``` for Cifar-10 are provided in the ```release``` section of the repository.

## Contact

Please contact Mitch Hill (mitchell.hill@ucf.edu) or Jonathan Mitchell (jcmitchell@ucla.edu) for any inquiries.
