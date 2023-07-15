# BiDAF with Answer-Pointer networks
An implementation of the [BiDAF](https://arxiv.org/abs/1611.01603) paper (with character level embeddings) for reading comprehension on the SqUAD v2 dataset. Also implemented an Answer Pointer network as per [Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905) to predict the final answer span. 

## Introduction

## Implementation details
* Only 325 of the total 442 articles in the SQuAD training dataset are used due to memory constraints.
* The answer pointer head uses dropout on the projected start pointer representation.
* A cosine learning rate scheduler is used.
* Hidden vectors in the BiDAF model are of 100 dimensions.
  
## Results


## Usage

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.

## References
    
