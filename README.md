# Neural Architecture Search for Multi-Task Learning Networks

This repository contains the implementation and experiments for my PhD thesis on Neural Architecture Search (NAS) applied to Multi-Task Learning (MTL) networks.

## Abstract

Multi-task learning (MTL) is a design paradigm for neural networks that aims to improve generalization while solving multiple tasks simultaneously in a single network. This research investigates the application of Neural Architecture Search (NAS) to automatically design MTL networks, comparing their performance against traditional single-task and hand-crafted multi-task networks.

The experiments cover various datasets including:
- ICAO-FVC
- MNIST
- FASHION-MNIST
- Celeb-A
- CIFAR-10

Key findings demonstrate that our Reinforcement Learning-based NAS technique can discover optimal architectures more efficiently than current state-of-the-art Regularized Evolution methods, while maintaining competitive performance across multiple MTL datasets.

## Project Structure

```
├── analysis/         # Notebooks for data analysis
├── preprocessing/    # Scripts and notebooks for preprocessing the datasets
├── src/              # Common source code used across many notebooks
├── training/         # Notebooks for NAS, MTL and STL experiments
└── vsoft/            # Comparison with Vsoft systems
```

## Key Features

- Implementation of Multi-Task Learning networks
- Neural Architecture Search using Reinforcement Learning
- Comparative analysis with:
  - Single-task networks
  - Hand-crafted multi-task networks
  - NAS-generated architectures
- Benchmarking on multiple standard datasets
- Performance evaluation metrics including accuracy and equal error rate

## Getting Started

### Installation

```bash
git clone https://github.com/guilhermemg/icao_nets_training
cd icao_nets_training
pip install -r requirements.txt
```

## Experiments

The repository includes experiments for:
1. Comparison of MTL vs single-task networks
2. NAS-generated architectures evaluation
3. Performance benchmarking across different datasets
4. Reinforcement Learning vs Regularized Evolution for NAS

## Results

Our experiments show that:
- RL-based NAS discovers optimal architectures faster than Regularized Evolution
- NAS-generated architectures achieve competitive results across various MTL datasets
- While not always outperforming hand-crafted architectures (e.g., in ICAO-FVC), the method shows promise for discovering potentially superior architectures

## Citation

If you use this code in your research, please cite:

    @phdthesis{gadelha2024investigation,
        author = {Gadelha, Guilherme Monteiro},
        title = {An Investigation of Neural Architecture Search in the Context of Deep Multi-Task Learning},
        school = {Federal University of Campina Grande},
        year = {2024},
        month = {January},
        address = {Campina Grande, PB, Brazil},
        note = {Advisors: Leonardo Vidal Batista and Herman Martins Gomes},
        type = {Ph.D. Thesis}
    }

## Acknowledgments

This research was supported by the National Council for Scientific and Technological Development (CNPq), the Laboratory of Artificial Intelligence and Dedicated Architectures (LIAD), and Vsoft Tecnologia Ltda. 
Thanks to my advisors, family, friends, and colleagues for their continuous support throughout this journey.
