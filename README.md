# Hate Speech and Offensive Language Detection


## Table of contents

* [Overview](#overview)
* [Goals](#goals)
* [Getting Started](#getting-started)
* [Running The Experiments](#running-the-experiments)

## Overview

This repository includes the utilities to test the performance of both shallow machine learning algorithms and deep learning models for the usecase of HSOLD. This task is being carried out as part of a study in NLP and should be treated as such.

## Goals

The objective of this project is to test and compare the performance of Traditional ML and DL models as well BERT when dealing with two datasets: the original, unbalanced HSOLD dataset, and a reduced balanced one, derived from the original one using random under-sampling.


## Getting Started

1. Clone repo

```
$ git clone git@github.com:DimitrisPatiniotis/Hate-Speech-and-Offensive-Language-Detection.git
```

2. Create a virtual environment and install all requirements listed in requirements.txt

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running The Experiments

To run the experiments simply execute main.py in the Processes folder

```
$ cd Processes/
$ python main.py
```