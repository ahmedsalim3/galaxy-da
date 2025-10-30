Domain Adaptation In Galaxy Morphology
======================================

Domain Adaptation for Galaxy Morphology Classification using llustrisTNG and Galaxy Zoo Evolution dataset

## Prerequisites

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Installation

1. Clone this [`repo`](https://github.com/ahmedsalim3/iaifi-hackathon-2025.git)

```bash
git clone https://github.com/ahmedsalim3/iaifi-hackathon-2025.git
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
pip install -e .
```

# How to train?

1. Create a config file, see [template](./configs/config.template.yml) and run with 

```sh
python3 scripts/run_train.py --config /path/to/config.yml
```

```sh
# Full script (Under Dev)
usage: run_train.py [-h] --config CONFIG [--device DEVICE] [--resume RESUME] [--use_diagnostics] [--eval_interval EVAL_INTERVAL]
                    [--diag_max_batches DIAG_MAX_BATCHES]

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --device DEVICE
  --resume RESUME
  --use_diagnostics
  --eval_interval EVAL_INTERVAL
  --diag_max_batches DIAG_MAX_BATCHES

See configs/config.template.yml for info about the config
```

## About This Project

This project was made possible through the [2025 IAIFI Summer School](https://github.com/iaifi/summer-school-2025) provided by The [NSF AI](https://iaifi.org/) Institute for Artificial Intelligence and Fundamental Interactions (IAIFI).

### Team Members

- [@ahmedsalim3](https://ahmedsalim3.github.io/)
- [@Meet-Vyas-Dev](https://meet-vyas-dev.github.io/)
- [@kaleybrauer](https://www.kaleybrauer.com/)
- [@adityadash54](https://github.com/adityadash54)
- [@stivenbg](https://github.com/stivenbg)
- [@dingq1](https://github.com/dingq1)
- [@AumRTrivedi](https://github.com/AumRTrivedi)

### References
