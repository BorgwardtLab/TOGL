# Topological Graph Neural Networks

This repository contains the code for our ICLR 2022 paper &ldquo;Topological
Graph Neural Networks.&ldquo; This repository requires either Python 3.7 or 3.8 to
be installed.

**Please note**: This repository is a work in progress!

## Citation

If you use this code, please consider citing our paper:

```bibtex
@InProceedings{Horn22a,
  author        = {Horn, Max and {De Brouwer}, Edward and Moor, Michael and Moreau, Yves and Rieck, Bastian and Borgwardt, Karsten},
  title         = {Topological Graph Neural Networks},
  year          = {2022},
  booktitle     = {International Conference on Learning Representations~(ICLR)},
  url           = {https://openreview.net/pdf?id=oxxUMeFwEHd},
}
```

## Quickstart

```bash
# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# Create venv and install dependencies of project
poetry install
# In case any errors occur, please rerun `poetry install`
poetry run install_deps_cpu  # or of course install_deps_cuXXX for GPU support
# Train TopoGNN on DD for 10 epochs
poetry run python topognn/train_model.py --model TopoGNN --dataset DD --max_epochs 10
```

## Installation

This repository uses `poetry` in order to manage dependencies and provide
a virtual environment for the project.  In order to install `poetry` please
refer to the instructions [here](https://python-poetry.org/docs/#installation).
After poetry is installed it can be used to create a virtual environment and
install the dependencies of the repository:

```bash
$ poetry install
# If any errors occur please rerun the command `poetry install`
```

Due to some incompatibilities between `poetry` and `torch_geometric` (which was
used for the implementation of GNNs in this repository), additional
dependencies need to be installed in a separate step dependent on the necessity
of CUDA based GPU acceleration.  This can be done using the command below

```bash
$ poetry run install_deps_{cpu, cu101, cu102, cu110}
```

where `{cpu, cu101, cu102, cu110}` should be replaced with either `cpu` or the
string matching the installed cuda toolkit version.

## Training models

The repository implements two models `TopoGNN` and `GNN`.  Additional
parameters can be passed to the script depending on the model and dataset
selected. For example, the `GNN` model and the `MNIST` dataset have the
following configuration options:

```bash
$ poetry run topognn/train_model.py --model GNN --dataset MNIST --help
usage: train_model.py [-h] [--model {TopoGNN,GCN}]
                      [--dataset {IMDB-BINARY,REDDIT-BINARY,REDDIT-5K,PROTEINS,PROTEINS_full,ENZYMES,DD,MUTAG,MNIST,CIFAR10,PATTERN,CLUSTER,Necklaces,Cycles,NoCycles}]
                      [--training_seed TRAINING_SEED]
                      [--max_epochs MAX_EPOCHS] [--paired PAIRED]
                      [--merged MERGED] [--logger {wandb,tensorboard}]
                      [--gpu GPU] [--hidden_dim HIDDEN_DIM]
                      [--lr LR]
                      [--dropout_p DROPOUT_P] [--GIN GIN]
                      [--train_eps TRAIN_EPS] [--batch_norm BATCH_NORM]
                      [--residual RESIDUAL] [--batch_size BATCH_SIZE]
                      [--use_node_attributes USE_NODE_ATTRIBUTES]

optional arguments:
  -h, --help            show this help message and exit
  --model {TopoGNN,GCN}
  --dataset {IMDB-BINARY,REDDIT-BINARY,REDDIT-5K,PROTEINS,PROTEINS_full,ENZYMES,DD,MUTAG,MNIST,CIFAR10,PATTERN,CLUSTER,Necklaces,Cycles,NoCycles}
  --training_seed TRAINING_SEED
  --max_epochs MAX_EPOCHS
  --paired PAIRED
  --merged MERGED
  --logger {wandb,tensorboard}
  --gpu GPU
  --hidden_dim HIDDEN_DIM
  --lr LR
  --dropout_p DROPOUT_P
  --GIN GIN
  --train_eps TRAIN_EPS
  --batch_norm BATCH_NORM
  --residual RESIDUAL
  --batch_size BATCH_SIZE
  --use_node_attributes USE_NODE_ATTRIBUTES
```

### Logging

By default runs are logged using Tensorboard, yet logging via WandB is also
possible. For this some adaptations to the code might be necessary in order to
log to the correct entity/project.  The logs of Tensorboard are by default
stored under the path `logs/{MODEL}_{DATASET}`.

### TopoGNN variants and ablations

In our work we show multiple variants and ablations of the TOGL layer. In
particular we present the `simple` variant, which does not actually perform any
topological computation, but merely includes all other components of the layer.
This setting can be triggered using the additional parameter `--fake True`.
Further, we differentiate between TOGL with coordinatization functions with the
parameter `--deepset False` and TOGL using a deep sets `--deepset True`.  Below
you can see some configurations which reflect the scenarios covered in our
work.

```bash
# TopoGNN with coord functions
poetry run topognn/train_model.py --model TopoGNN --dataset DD --batch_size 20 --lr 0.0007

# GCN on DD
poetry run topognn/train_model.py --model GCN --dataset DD --batch_size 20 --lr 0.0007
```

Additional calls are also possible; stay tuned for more details or check
out available models in `models.py`.

### Synthetic datasets
The synthetic datasets used to train our models are provided in the folder
`data/SYNTHETIC`. If these are not compatible with your architecture (which
could happen as they are saved in a binary file format), you can regenerate the
synthetic datasets using calls to the script `data/SYNTHETIC/datagen.py`:
```bash
cd data/SYNTHETIC
poetry run python datagen.py --dataset Cycle --min_cycle 3
poetry run python datagen.py --dataset Necklaces
```

## Results used for plots
The results and data used to plot our figures can be found in the folder
`plots_data` and code to generate plots can be found in the `notebooks` folder.
