# Model Shuffle Attack (MSA)

This repository contains example code for the Model Shuffle Attack (MSA). MSA is a technique used to demonstrate vulnerabilities in federated learning systems, particularly focusing on model poisoning attacks.

## Overview

Federated learning is a machine learning paradigm that enables training models across multiple decentralized devices while preserving user privacy. However, it is susceptible to various attacks, including model poisoning. The Model Shuffle Attack aims to exploit these vulnerabilities by introducing malicious updates during the training process.

This project is based on the implementation from [https://github.com/shaoxiongji/federated-learning](https://github.com/shaoxiongji/federated-learning).

For more details, refer to the original paper: [Model poisoning attack in differential privacy-based federated learning](https://www.sciencedirect.com/science/article/abs/pii/S0020025523002141).

## Features

- Implementation of Model Shuffle Attack in a federated learning setting.
- Demonstration of attack impact on model accuracy and convergence.
- Configurable parameters for attack strength.

## Requirements

- Python 3.9.20
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/paranoid-27/MSA.git
   cd MSA
2. Install the required packages:
	```bash
	pip install -r requirements.txt

## Usage

To run the MSA simulation, use the following command:
	```bash

	python main_fed_cifar_cnn.py

## Configuration
The configuration for the federated learning setup and the Model Shuffle Attack (MSA) parameters are specified in the `utils/options_cifar.py` file.

This script contains all the necessary configurations for running experiments with CIFAR datasets and simulating the Model Shuffle Attack.

## Results

The results of the Model Shuffle Attack simulation, including metrics such as model accuracy and loss, are tracked and visualized using [Weights & Biases](https://wandb.ai/). This tool provides an interactive dashboard for analyzing the impact of the attack on model performance.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Original implementation: https://github.com/shaoxiongji/federated-learning

Original paper: Model poisoning attack in differential privacy-based federated learning

## Citation

If you find this project helpful or use it in your research, please consider citing this paper:

Yang M, Cheng H, Chen F, et al. Model poisoning attack in differential privacy-based federated learning[J]. Information Sciences, 2023, 630: 158-172.

You can use the following BibTeX entry for citation:

```bibtex
@article{yang2023model,
  title={Model poisoning attack in differential privacy-based federated learning},
  author={Yang, M and Cheng, H and Chen, F and others},
  journal={Information Sciences},
  volume={630},
  pages={158--172},
  year={2023},
  publisher={Elsevier}
}
