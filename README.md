# COMP6248 Reproducibility Project

## About

This report analyses the reproducibility of paper on [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768).
One of the challenges in machine learning research is to ensure that published results are reliable and reproducible. In support of this, the objective of this challenge is to investigate reproducibility of empirical results submitted to NeurIPS. We reimplement the methods described by the paper to compare results on the same dataset.


## Implementation
Wide Residual Nets (WRNs) are used for both the teacher and the student networks in the few-shot and zero-shot algorithms. For the implementation of such networks, we opted to use the authors' code with slight adaptations. Everything was implemented on our own including Zero-Shot, KD+AT and adversarial belief matching.


## Experimentation
To replicate the results form the original paper, the CIFAR10 and SVHN datasets are used for the experiments. For comparison, a WRN trained with the subset of images and labels (no teacher) and a few-shot model learnt with the full data (KD-AT full data) are compared to the few-shot trained with the down-sampled dataset (KD+AT) and the zero-shot model. Keeping inline with the original paper, the few-shot model is referred to as KD-AT (named after the knowledge distillation and attention transfer loss function). For each dataset, the KD-AT and No Teacher models are trained with the downsampled datasets with M images per class where M ∈ {10, 25,50, 75, 100}. The KD+AT full data is trained once only using the full dataset and hence its test accuracy is the same across all values of M. To generate the plots the teacher and student model sizes used were WRN-40-2 (depth 40 and widen factor 2) and WRN-16-1 (depth 16 and widen factor 1), respectively.

### Enviroment

- Python 3.6+
- Scipy
- torchbearer
- PyTorch

Install all python modules with

```
pip install -r requirements.txt
```
or if you have different versions of Python installed:
```
pip3 install -r requirements.txt
```

## Train from scratch

To run the models, configure [config.py](./src/config.py) tweaking the parameters as required. Then run
```
python src/main.py
```
or
```
python3 src/main.py
```


## Acknowledgements
This work was initiated as a project of our master's level course 'COMP6248: Deep Learning' at University of Southampton. We would like to thank the course staff for providing us with the necessary GPU hardware to use during training phase.


## Built With

* [Pytorch](https://pytorch.org/) - Machine learning framework

## Authors

* **Alex Newton** - [xandernewton](https://github.com/xandernewton)
* **Diogo Filipe Pinto Pereira** - [DiogoP98](https://github.com/DiogoP98)
* **Subash Poudyal** - [subash774](https://github.com/subash774)
