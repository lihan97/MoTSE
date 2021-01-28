# MoTSE
Codes for "MoTSE: an interpretable task similarity estimator for small molecular property prediction tasks"

## Environments
Tested on a linux server with GeForce GTX 1080 and the running environment is as follows:
*python=3.6

*pytorch=1.1.0

*dgl=0.4.2

*rdkit=2018.09.3


## File Description
*./datasets/: This directory contains QM9 dataset, PCBA dataset, and Alchemy dataset used in paper.

*./src/models.py: This file contains the codes of graph convolutional network.

*./src/motse.py: This file contains the codes of our model.

*./src/trainer.py: This file contains the codes of the trainer to train the graph convolutional network.

*./src/utils/: This directory contains the codes of utils used in the experiments.

*./src/1.train.ipynb: This is a demo for training GCN on QM9 dataset.

*./src/2.transfer.ipynb: This is a demo for transfer learning on QM9 dataset.

*./src/3.estimate_similarity.ipynb: This is a demo for similarity estimation on QM9 dataset.

*./src/4.evaluate.ipynb: This is a demo for evaluating on QM9 dataset.



## License
This software is copyrighted by Machine Learning and Computational Biology Group @ Tsinghua University.

The algorithm and data can be used only for NON COMMERCIAL purposes.
