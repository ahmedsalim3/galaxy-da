From Simulations to Surveys: Domain Adaptation for Galaxy Observations
======================================================================

<div align="center">

[Paper](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_265.pdf) | [arXiv](https://arxiv.org/abs/2511.18590) | [Dataset](https://zenodo.org/records/17434016) | [Page](https://ahmedsalim3.github.io/galaxy-da) | [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/123052.png)

</div>

<div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px;">
    <img src="./paper-figures/figure2_training.png" alt="train-log" style="width: 100%; height: 90%;">
  </div>
  <div style="flex: 1; min-width: 300px;">
    <img src="./paper-figures/figure6_ot_loss.png" alt="ot-loss" style="width: 100%; height: 90%;">
  </div>
</div>

## Abstract

Large photometric surveys will image billions of galaxies, but we currently lack quick, reliable automated ways to infer their physical properties like morphology, stellar mass, and star formation rates. Simulations provide galaxy images with ground-truth physical labels, but domain shifts in PSF, noise, backgrounds, selection, and label priors degrade transfer to real surveys. We present a preliminary domain adaptation pipeline that trains on simulated TNG50 galaxies and evaluates on real SDSS galaxies with morphology labels (elliptical/spiral/irregular). We train three backbones (CNN, $E(2)$-steerable CNN, ResNet-18) with focal loss and effective-number class weighting, and a feature-level domain loss $\mathcal{L}_D$ built from [GeomLoss](./geomloss/) (entropic Sinkhorn OT, energy distance, Gaussian MMD, and related metrics). We show that a combination of these losses with an OT-based "top-$k$ soft matching" loss that focuses $\mathcal{L}_D$ on the worst-matched source–target pairs can further enhance domain alignment. With Euclidean distance, scheduled alignment weights, and top-$k$ matching, target accuracy rises from ~61% (no adaptation) to ~86–89%, with a ~17-point gain in macro–F1 and a domain AUC near 0.5, indicating strong latent-space mixing.

![latent-space](./paper-figures/figure3_latent_space.png)

![cm-metrics](./paper-figures/figure4_cm_and_metrics.png)

## Prerequisites

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Installation

1. Clone this [`repo`](https://github.com/ahmedsalim3/galaxy-da.git)

```bash
git clone https://github.com/ahmedsalim3/galaxy-da.git
```

2. Install dependencies:

```bash
make install
```

- or you can create a `.venv`:

```sh
python3 -m venv .venv
source .venv/bin/activate # On mac/linux distros
```
- Install `nebula`

```sh
git submodule update --init --recursive
pip install -e .
```

## Data

To access the datasets used in this project, please refer to the [Zenodo dataset repository](https://zenodo.org/records/17434016). The dataset includes RGB galaxy images and labels for both source (IllustrisTNG) and target (SDSS, Galaxy Zoo 2) domains.

## How to train?

1. Create a config file, see [template](./configs/config.template.yml) and run with 

```sh
python3 scripts/run_train.py --config /path/to/config.yml
```

## How to evaluate?

```sh
python3 scripts/run_eval.py /path/to/ckpt
```
You can also run train and evaluate simultaneously. Run this with a single config, multiple configs, or a folder of configs by passing `-f`:

```sh
./run_experiment.sh <config_path> [more_configs...]
# or for a folder of configs:
./run_experiment.sh -f <config_folder>
```

## Experiments

The [experiments](./experiments/README.md) directory contains the paper experiments. For other experiments, checkout to the [distance-metrics branch](https://github.com/ahmedsalim3/galaxy-da/tree/distancemetrics_branch) or [ot-alignment](https://github.com/ahmedsalim3/galaxy-da/tree/ot-alignment) branch:

## Paper Figures

The [paper-figures](./paper-figures/) directory contains all the figures used in the paper. To reproduce the paper plots, run:

```sh
python3 scripts/make_paper_figures.py experiments
```


## Citation

```
@misc{brauer2025simulationssurveysdomainadaptation,
      title={From Simulations to Surveys: Domain Adaptation for Galaxy Observations}, 
      author={Kaley Brauer and Aditya Prasad Dash and Meet J. Vyas and Ahmed Salim and Stiven Briand Massala},
      year={2025},
      eprint={2511.18590},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA},
      url={https://arxiv.org/abs/2511.18590}, 
}
```

## About This Project

This project was made possible through the [2025 IAIFI Summer School](https://iaifi.org/phd-summer-school.html) provided by The [NSF AI](https://iaifi.org/) Institute for Artificial Intelligence and Fundamental Interactions (IAIFI). This work will be presented at the Machine Learning and the Physical Sciences Workshop @ [NeurIPS 2025](https://ml4physicalsciences.github.io/2025/)
