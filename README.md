# MoTSE
Codes for "MoTSE: an interpretable task similarity estimator for small molecular property prediction tasks"

## Environments
Tested on a linux server with GeForce GTX 1080.

Setup the environment using environment.yml with Anaconda. While in the project directory run:

conda env create

Activate the environment

conda activate MoTSE

## How to Run
1. Run ./src/1.train.ipynb for training GCN models on QM9 dataset.

2. Run ./src/2.transfer.ipynb for transfer learning on QM9 dataset.

3. Run ./src/3.estimate_similarity.ipynb for estimating similarity of tasks in QM9 dataset.

4. Run ./src/4.evaluate.ipynb for evaluating the performance on QM9 dataset.

The expected output had been displayed in the ipynb files. And it takes around 5 hours to run these four steps.

## File Description
* ./datasets/: This directory contains QM9 dataset and PCBA dataset used in paper.

* ./src/models.py: This file contains the codes of graph convolutional network.

* ./src/motse.py: This file contains the codes of our model.

* ./src/trainer.py: This file contains the codes of the trainer to train the graph convolutional network.

* ./src/utils/: This directory contains the codes of utils used in the experiments.

* ./src/1.train.ipynb: This is a demo for training GCN on QM9 dataset.

* ./src/2.transfer.ipynb: This is a demo for transfer learning on QM9 dataset.

* ./src/3.estimate_similarity.ipynb: This is a demo for similarity estimation on QM9 dataset.

* ./src/4.evaluate.ipynb: This is a demo for evaluating on QM9 dataset.
