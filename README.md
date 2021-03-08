# UstarClassification
A simple demo for the paper "Binary Classification from Multiple Unlabeled Datasets via Surrogate Set Classification"

## Requirements
+ Python 3
+ Numpy 1.19.2
+ Keras 2.4.3
+ Tensorflow 2.2.0
+ Matplotlib 3.3.4
+ Seaborn 0.11.1

## A Demo for MNIST dataset
You can run an example code of U^m-SSC method on MNIST dataset and spcify the number of U sets.

`python experiment.py --dataset mnist --sets 20`

The output will be in the folder ./output_SSC/mnist/m,
where m is the number of set of this run.
