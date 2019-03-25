# Final Year Project Code

This repository contains the implementation of the graph neural networks implemented for the Final Year Project (FYP). It includes models for [Gated Graph Neural Networks (GGNN)](https://arxiv.org/abs/1511.05493) and [Residual Gated Graph ConvNets (RGGC)](https://arxiv.org/abs/1711.07553). This was originally forked from [JamesChuanggg/ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch) and modified to include the RGGC model [here](https://github.com/xbresson/spatial_graph_convnets). Both models are tested against the [bAbi tasks dataset](https://research.fb.com/downloads/babi/). Here's an example of bAbI deduction task (task 15):

<img src="images/babi15.png" width=700>

The blog for this project can be found [here](https://calebmah.me/fyp-questions-and-answers/).

## Requirements
- python==3.7
- PyTorch>=1.0
- XlsxWriter>=1.1.5

## Installation
Using conda environments:
```
conda env create -f environment.yml
activate fyp
```

## Run
View help:
```
python main.py --help
```

Suggesting configurations for each task (GGNN):
```
# task 1
python main.py --net "GGNN" --task_id 1 --lr 0.005 --state_dim 8 --n_steps 5 --niter 100 --cuda
# task 2
python main.py --net "GGNN" --task_id 2 --lr 0.005 --state_dim 4 --n_steps 5 --niter 100 --cuda
# task 4
python main.py --net "GGNN" --task_id 4 --lr 0.05 --state_dim 10 --n_steps 10 --niter 100 --cuda
# task 9
python main.py --net "GGNN" --task_id 9 --lr 0.005 --state_dim 8 --n_steps 5 --niter 100 --cuda
# task 11
python main.py --net "GGNN" --task_id 11 --lr 0.005 --state_dim 8 --n_steps 10 --niter 100 --cuda
# task 12
python main.py --net "GGNN" --task_id 12 --lr 0.01 --state_dim 8 --n_steps 5 --niter 100 --cuda
# task 13
python main.py --net "GGNN" --task_id 13 --lr 0.005 --state_dim 8 --n_steps 5 --niter 100 --cuda
# task 15
python main.py --net "GGNN" --task_id 15 --lr 0.05 --state_dim 4 --n_steps 10 --niter 100 --cuda
# task 16
python main.py --net "GGNN" --task_id 16 --lr 0.01 --state_dim 8 --n_steps 5 --niter 100 --cuda
# task 17
python main.py --net "GGNN" --task_id 17 --lr 0.005 --state_dim 4 --n_steps 10 --niter 100 --cuda
# task 18
python main.py --net "GGNN" --task_id 18 --lr 0.005 --state_dim 8 --n_steps 5 --niter 100 --cuda
```

Suggesting configurations for each task (RGGC):
```
# task 1
python main.py --net "RGGC" --task_id 1 --lr 0.05 --state_dim 10 --n_steps 5 --niter 100 --cuda
# task 2
python main.py --net "RGGC" --task_id 2 --lr 0.05 --state_dim 8 --n_steps 10 --niter 100 --cuda
# task 4
python main.py --net "RGGC" --task_id 4 --lr 0.01 --state_dim 8 --n_steps 10 --niter 100 --cuda
# task 9
python main.py --net "RGGC" --task_id 9 --lr 0.005 --state_dim 8 --n_steps 5 --niter 100 --cuda
# task 11
python main.py --net "RGGC" --task_id 11 --lr 0.01 --state_dim 10 --n_steps 10 --niter 100 --cuda
# task 12
python main.py --net "RGGC" --task_id 12 --lr 0.005 --state_dim 4 --n_steps 10 --niter 100 --cuda
# task 13
python main.py --net "RGGC" --task_id 13 --lr 0.005 --state_dim 4 --n_steps 5 --niter 100 --cuda
# task 15
python main.py --net "RGGC" --task_id 15 --lr 0.05 --state_dim 10 --n_steps 10 --niter 100 --cuda
# task 16
python main.py --net "RGGC" --task_id 16 --lr 0.01 --state_dim 4 --n_steps 5 --niter 100 --cuda
# task 17
python main.py --net "RGGC" --task_id 17 --lr 0.01 --state_dim 4 --n_steps 5 --niter 100 --cuda
# task 18
python main.py --net "RGGC" --task_id 18 --lr 0.005 --state_dim 10 --n_steps 10 --niter 100 --cuda
```

## Results
Only 50 randomly selected training examples for training for each task.
Performances are evaluated on 50 random examples.

| bAbI Task | GGNN Accuracy | RGGC Accuracy |
| ------| ------ | ------ |
| 1 | 46.00% | 47.60% |
| 2 | 36.00% | 37.60% |
| 4 | 100.00% | 58.95% |
| 9 | 8.00% | 38.40% |
| 11 | 31.20% | 35.20% |
| 12 | 28.00% | 30.40% |
| 13 | 27.60% | 28.40% |
| 15 | 100.00% | 73.50% |
| 16 | 100.00%| 100.00% |
| 17 | 29.67% | 43.96% |
| 18 | 33.86% | 28.35% |

## Disclaimer
The data processing codes are from official implementation [yujiali/ggnn](https://github.com/yujiali/ggnn).
