# From Simulations to Surveys: Domain Adaptation for Galaxy Observations

---

## Abstract

Large photometric surveys will image billions of galaxies, but we currently lack quick, reliable automated ways to infer their physical properties like morphology, stellar mass, and star formation rates. Simulations provide galaxy images with ground-truth physical labels, but domain shifts in PSF, noise, backgrounds, selection, and label priors degrade transfer to real surveys. We present a preliminary domain adaptation pipeline that trains on simulated TNG50 galaxies and evaluates on real SDSS galaxies with morphology labels (elliptical/spiral/irregular). We train three backbones (CNN, $E(2)$-steerable CNN, ResNet-18) with focal loss and effective-number class weighting, and a feature-level domain loss $\mathcal{L}_D$ built from GeomLoss (entropic Sinkhorn OT, energy distance, Gaussian MMD, and related metrics). We show that a combination of these losses with an OT-based "top-$k$ soft matching" loss that focuses $\mathcal{L}_D$ on the worst-matched source–target pairs can further enhance domain alignment. With Euclidean distance, scheduled alignment weights, and top-$k$ matching, target accuracy rises from ~61% (no adaptation) to ~86–89%, with a ~17-point gain in macro–F1 and a domain AUC near 0.5, indicating strong latent-space mixing.

---

## About

This project was made possible through the [2025 IAIFI Summer School](https://github.com/iaifi/summer-school-2025) provided by The [NSF AI](https://iaifi.org/) Institute for Artificial Intelligence and Fundamental Interactions (IAIFI). This work was presented at the Machine Learning and the Physical Sciences Workshop @ [NeurIPS 2025](https://neurips.cc/)
