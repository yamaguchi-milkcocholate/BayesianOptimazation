
# IGO module

## Update information

### version 0.2.1
* Modify Readme.md
* Tidy up switching the problem type (minimization or maximization) ('objective_function/base.py', 'util/weight.py')
* Skip the square root calculation when negative eigenvalues occur ('util/model.py')
* Tidy up the tie case averaging in CMAWeight ('util/weight.py')
* Update the quantile estimation for computational efficiency ('util/weight.py')

### version 0.2
* Add CMA-ES and Reuse-CMA algorithms
* Move the benchmark function implementation to the directory 'objective_function'.
* Separate the model definition ('util/model.py') and the model update (in 'optimizer' dir) codes.
* Modify the directory structure
* Fix the Bohachevsky function
* Modify log-likelihood calculation

## Usage
* Please see 'sample_script.py'

## Directory structure
```
igo
├── objective_function # benchmark function
├── optimizer # model update and optimization
└── util # weight, model definition, sampling method, log, etc
```

## Requirement
We tested on the following environment:
* Python 2.7.13 :: Anaconda 4.3.0 (x86_64)

## Todo
### General
* Python 3 support
* Maintenance of the document (manual)

### Model
* Categorical distribution
* Diagonal gaussian
* Upper triangle (A) parameterization of the covariance matrix

### Algorithm
* Two point step size adaptation
* Lebesgue based weights
* (1 + 1)-CMA-ES
* Normalization of the covariance matrix
* Recalculation of the learning rate on ReuseIGO

