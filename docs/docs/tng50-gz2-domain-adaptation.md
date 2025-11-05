Introduction
============

Covariate shift happens when data distribution changes between the dataset used during training and
those outside of it, often not just image properties but class distribution as well. For example, different sensors and different sampling bias, but the conditional distribution of labels given the data remains the same. We study whether domain adaptation techniques can help us narrow this gap in a realistic setting: training a classification task on mock observations of simulated galaxies from TNG50 SKIRT and testing on real observations from the Galaxy Zoo 2 (GZ2) project. Having them both consisting of three classes, "elliptical", "irregular", "spiral"

The source dataset consists of mock galaxy observations generated from the IllustrisTNG (TNG50) cosmological simulation using the SKIRT radiative transfer code. We select galaxies with reliable morphology and photometric properties, convolve the images with Gaussian PSF, and add realistic sky noise to approximate observational conditions. Morphology labels are derived from the simulation’s intrinsic physical parameters, yielding 3,232 multi-band images used for model training. 

Despite the large number of examples, the class distribution is highly imbalanced, of which 2,305 are spiral galaxies, 719 are elliptical, and only 208 are irregular. This corresponds to roughly 71 % spirals, 22 % ellipticals and 6 % irregulars. Such skewed frequencies motivate the use of class weighting and focal loss in the classification objective

The target dataset, on the other hand, is constructed from the GZ2 project [2], using the debiased vote
fractions provided by [Hart et al. (2016)][gz2hart-paper] to correct for redshift and observational biases. Real galaxy observations are obtained from the Sloan Digital Sky Survey (SDSS), and morphology labels are assigned by applying strict thresholds to debiased vote fractions $p_{\text{debiased}}$ to ensure high-confidence classifications and to match the source data labels. Galaxies are classified into three morphology classes:

| Class | Conditions |
|:------|:------------|
| **elliptical** | $$p_{\text{smooth}} \geq \tau_{\text{ell}}, \quad p_{\text{nospiral}} \geq \tau_{\text{nospiral}}, \quad p_{\text{edgeon}} < \tau_{\text{edgeon,ell}}$$ |
| **spiral** | $$ p_{\text{spiral}} \geq \tau_{\text{spiral}}, \quad p_{\text{features}} \geq \tau_{\text{features}}, \quad p_{\text{edgeon}} < \tau_{\text{edgeon,spiral}}$$ |
| **irregular** | $$p_{\text{odd}} \geq \tau_{\text{odd}}, \quad \max(p_{\text{irregular}}, p_{\text{merger}}, p_{\text{disturbed}}) \geq \tau_{\text{irregular}}$$ |

- where $\tau$ denotes the respective thresholds (e.g., $\tau_{\text{ell}} = 0.95$, $\tau_{\text{spiral}} = 0.95$).

- Galaxies with $p_{\text{artifact}} \geq \tau_{\text{artifact}}$ are filtered out, resulting in an ultra-clean dataset with high-confidence morphology labels.


In addition to morphology classification, the dataset includes supplementary labels that enable potential extensions to **multi-task learning** frameworks. Stellar mass values $M_\ast$ (in solar masses) are derived from cross-matched AGN catalogs by [Schawinski et al. (2010)](https://adsabs.harvard.edu/abs/2010ApJ...711..284S) and computed via

$$
M_\ast = 10^{\log_{10} M_\ast^{\text{catalog}}}
$$

where $\log_{10} M_\ast^{\text{catalog}}$ is the logarithmic stellar mass from the catalog. These mass estimates could serve as a **regression target** $y_{\text{mass}} \in \mathbb{R}^+$ for predicting galaxy mass from images. Binary classification targets include **star formation** flags $y_{\text{SF}} \in \{0, 1\}$ (derived from Baldwin–Phillips–Terlevich diagram classifications, where $y_{\text{SF}} = 1$ if BPT_CLASS $= 1$ for pure star-forming galaxies) and **AGN presence** flags $y_{\text{AGN}} \in \{0, 1\}$ (where $y_{\text{AGN}} = 1$ if BPT_CLASS $\in \{3, 4\}$ for Seyfert and LINER types). These additional labels, cross-matched from the catalogs by [Lintott et al. (2008)](https://adsabs.harvard.edu/abs/2008MNRAS.389.1179L) and [Lintott et al. (2011)](https://adsabs.harvard.edu/abs/2011MNRAS.410..166L), provide a foundation for future multi-task models that simultaneously predict morphology $\mathbf{y}_{\text{morph}}$, mass $y_{\text{mass}}$, and physical properties $\mathbf{y}_{\text{phys}} = [y_{\text{SF}}, y_{\text{AGN}}]^\top$ of galaxies.


----

Problem Setup
=============

**Source domain dataset**:

$\mathcal{D}_S = \{(x_i^{(S)}, y_i^{(S)})\}_{i=1}^{n_S}$,
  - where $x_i^{(S)} \sim p_S(x)$ are galaxy morphology observations from TNG, and $y_i^{(S)} \in \{0, 1, 2\}$ are the corresponding morphology labels (elliptical, spiral, irregular, respectively). Here, $n_S$ labeled samples are drawn from the joint distribution $P_S(x, y)$.

**Target domain dataset**:  

$\mathcal{D}_T = \{x_i^{(T)}\}_{i=1}^{n_T}$,

  - where $x_i^{(T)} \sim p_T(x)$ are unlabeled galaxy morphology observations from the GZ2 Legacy Imaging Surveys. The included morphologies are the same in both datasets, with $n_T$ unlabeled samples drawn from the marginal distribution $P_T(x)$.

We assume the label distributions are aligned, i.e., $p_S(y|x) = p_T(y|x)$, but the input distributions differ: $p_S(x) \neq p_T(x)$. Our goal is to adapt $f_\theta$ using **only labeled source data** and **unlabeled target data**, so that it performs well on the target domain.


Neural Network
==================

Let $f_\theta$ denote a neural network classifier with parameters $\theta$, which maps an input image $x \in \mathbb{R}^{C \times H \times W}$ to class logits $\hat{y} = f_\theta(x) \in \mathbb{R}^K$, where $C=3$ is the number of channels, $H \times W$ are the spatial dimensions, and $K=3$ is the number of classes. The network is decomposed into a feature extractor $\phi_\theta : \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^d$ and a classifier head $g_\theta : \mathbb{R}^d \rightarrow \mathbb{R}^K$, such that $f_\theta(x) = g_\theta(\phi_\theta(x))$ outputs logits. Class probabilities are obtained via $\text{softmax}(f_\theta(x))$.


We define the **latent representation** $z = \phi_\theta(x) \in \mathbb{R}^d$ as the flattened feature output from the convolutional feature extractor (before the classifier head). The feature extractor consists of two convolutional blocks (64 and 128 channels) with batch normalization, ReLU activations, and max pooling, followed by flattening. The classifier head is a multi-layer perceptron with dimensions $d \rightarrow 256 \rightarrow 128 \rightarrow K$. This latent representation (also called the *latent vector* or *latent space*) will be used for domain alignment.

Architectures
-------------

### Models

We evaluate three encoders that all produce class logits $\hat{y}$ and a latent representation $z$:

- CNN: a compact convolutional network with two convolutional blocks and a small MLP head.
- ResNet‑18 (pretrained): an ImageNet‑pretrained backbone with a lightweight MLP head; fine‑tuning is restricted to the last stage(s).
- E(2)‑equivariant CNN: a steerable architecture with group convolutions (cyclic $C_N$ rotations or dihedral $D_N$ rotations+flips), equivariant normalization and nonlinearities, followed by group pooling and an MLP head.

In all cases the forward map returns $(\hat{y}, z)$, so the same domain‑adaptation losses and training protocol apply. For the equivariant model we do not duplicate rotated inputs (equivariance already accounts for orientations); for non‑equivariant baselines we include rotation augmentation.

### Classification loss

On the source domain, we minimize the supervised cross-entropy loss between predicted and true labels:

$$
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N_s} \sum_{i=1}^{N_s} \log f_\theta^{(y_s^{(i)})}(x_s^{(i)}),
$$

where $f_\theta^{(k)}(x)$ denotes the predicted probability for class $k$. In our three‑class morphology problem we take $k = 3$ with labels indexed as $\{0,1,2\}$ for “elliptical”, “irregular” and “spiral”, respectively. The class probabilities are obtained by applying the softmax function to the logits $f_\theta(x)$

Methods
=======


To stabilize the training process, we introduce an initial "warm-up" phase controlled by a a given parameter. During these initial epochs, the model is trained exclusively on the source domain data; the domain adaptation (DA) loss component is not computed. This phase is intended to predispose the model to an ideal location in the loss landscape before starting the adaptation objective. Furthermore, to prevent premature termination before adaptation begins, early stopping validation checks are suspended during this period. The duration of the warm-up phase was tuned for each experiment, ensuring it was long enough for the model to become performant on the source domain but short enough to avoid overfitting.


Losses
=====================

In our framework the total training objective is a weighted combination of a supervised classification loss on the source domain and an optional domain‑alignment loss computed on the latent representations. The classification component encourages the network to predict the correct morphology on labelled source examples, while the domain‑alignment term reduces the discrepancy between source and target feature distributions.

### Classification objectives

The default classification loss is the cross‑entropy defined above. Because the galaxy morphology classes are severely imbalanced, we optionally apply class weights or switch to focal loss to focus learning on minority classes. Given logits $h=f_\theta(x)$ and ground‑truth label $y\in \{0,1,2\}$, focal loss is

$$ \mathcal{L}_{\text{Focal}}(h,y) = -\alpha_y\,(1 - p_y)^{\gamma}\,\log p_y,\qquad p_y = \frac{\exp(h_y)}{\sum_k \exp(h_k)}, $$

where $\gamma\ge 0$ controls how strongly the loss focuses on misclassified examples and $\alpha_y$ is an optional per‑class weighting term. Setting $\gamma=0$ and $\alpha_y=1$ recovers the standard cross‑entropy. In the experiments we choose $\gamma\approx2$ and use either the effective number of samples or manually specified weights for $\alpha_y$, as indicated in the configuration files.

### Domain‑alignment losses

Let ${z_i^s}_{i=1}^{n_s}$ and ${z_j^t}_{j=1}^{n_t}$ denote the latent features of the source and target domains extracted by the encoder $\phi_\theta$. We consider several ways to encourage the distributions of $z_s$ and $z_t$ to match:

#### 1. Entropic optimal transport (Sinkhorn): 

The Sinkhorn divergence approximates the Wasserstein‑2 distance between two empirical distributions using entropic regularisation. Let $\pi \in \mathbb{R}^{n_s \times n_t}$ be a transport plan, $c_{ij} = |z_i^s - z_j^t|^2$ a cost matrix and $\sigma>0$ the blur parameter. The loss solves 

$$ 
W_{\sigma}(z_s,z_t) = \min_{\pi\in\Pi} \sum_{i,j} c_{ij}\,\pi_{ij} - \sigma H(\pi),
$$ 

where $H(\pi)=-\sum_{i,j}\pi_{ij}\log \pi_{ij}$ and $\Pi$ denotes the set of doubly stochastic matrices. As $\sigma\to0$ this recovers the unregularised Wasserstein distance; as $\sigma$ grows it approaches a maximum mean discrepancy.

In the [code](../nebula/modeling/domain_losses.py#L12) this is implemented via geomloss with given $\sigma$ parameter.


#### 2. Energy distance:

An energy distance (equivalent to the squared maximum mean discrepancy with a Laplacian kernel) is defined as $$ E(z_s,z_t) = 2\,\mathbb{E}_{i,j}[|z_i^s - z_j^t|] - \mathbb{E}_{i,i'}[|z_i^s - z_{i'}^s|] - \mathbb{E}_{j,j'}[|z_j^t - z_{j'}^t|]. $$ This loss encourages the average distance between latent features from different domains to match those within the same domain. It is implemented via SamplesLoss("energy", p=2) and does not depend on a blur parameter.


#### 3. Gaussian MMD: 

Maximum mean discrepancy (MMD) with a Gaussian kernel of bandwidth $\sigma$ compares the distributions via $$ \mathrm{MMD}_\sigma^2(z_s,z_t) = \mathbb{E}_{i,i'}[k_\sigma(z_i^s,z_{i'}^s)] + \mathbb{E}_{j,j'}[k_\sigma(z_j^t,z_{j'}^t)] - 2\,\mathbb{E}_{i,j}[k_\sigma(z_i^s,z_j^t)], $$ where $k_\sigma(u,v) = \exp\big(-|u-v|^2/(2\sigma^2)\big)$. Larger values of $\sigma$ yield smoother kernels and more emphasis on global structure. This is implemented via SamplesLoss("gaussian", blur=\sigma).
Sigma Scheduling (Sinkhorn)


#### 4. Domain adversarial loss: 

Instead of an explicit discrepancy metric, domain adversarial training introduces a binary domain classifier $d_\psi$ that predicts whether a latent vector comes from the source ($y=1$) or target ($y=0$) domain. A gradient reversal layer flips the sign of the gradient so that the feature extractor learns to fool the domain classifier while still minimizing the classification loss on the source. The domain adversarial loss is the binary cross‑entropy $$ \mathcal{L}_{\text{adv}} = -\frac{1}{n_s+n_t}\sum_{i=1}^{n_s+n_t} \big[y_i\,\log d_\psi(z_i) + (1-y_i)\,\log(1-d_\psi(z_i))\big], $$ and the total loss becomes $\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{adv}}$, where $\lambda$ controls the strength of gradient reversal.



When combining the supervised and domain‑alignment terms we either use a fixed weighting factor $\lambda_{\mathrm{DA}}$ (fixed‑lambda variant) or the homoscedastic uncertainty weighting proposed by Kendall et al. (2018). In the latter case the total loss for the classification and domain tasks is $$ \mathcal{L}_{\text{total}}(\eta_1,\eta_2) = \frac{1}{2\eta_1^2} \mathcal{L}_{\text{CE}} + \frac{1}{2\eta_2^2} \mathcal{L}_{\text{DA}} + \log(\eta_1 \eta_2), $$ where $\eta_1$ and $\eta_2$ are trainable positive parameters. The logarithmic term regularises the weights and prevents them from becoming arbitrarily small. After each optimisation step the values are clamped to avoid degenerate solutions.


### Sigma Scheduling (Sinkhorn)

The Sinkhorn divergence includes a blur (or entropy) parameter $\sigma$ that controls the trade‑off between computational efficiency and fidelity to the unregularised Wasserstein distance. Large blur values yield a loss that resembles a Gaussian MMD and is easier to optimise; small blur values approximate the true optimal transport cost but can be numerically unstable. To benefit from both regimes we anneal $\sigma$ over the course of training.
We denote by $\sigma_t$ the blur value used at epoch $t$ and by $T$ the total number of epochs. The code implements several scheduling strategies:

1. Exponential decay (default in our experiments). Starting from an initial blur $\sigma_0$ the blur decays geometrically: $\sigma_t = \sigma_0\,\delta^t$ with decay rate $0<\delta<1$. For example, $\sigma_0=10$ and $\delta=0.6$ means the blur drops from $10$ to $10 \times 0.6^5 \approx 0.78$ over five epochs.
    
2. Linear decay. The blur decreases linearly from $\sigma_0$ to a final blur $\sigma_{\mathrm{final}}$ according to $\sigma_t = \sigma_0 - (\sigma_0 - \sigma_{\mathrm{final}})\,t/(T-1)$.

3. Cosine annealing. A smooth schedule given by $\sigma_t = \sigma_{\mathrm{final}} + \tfrac{1}{2}(\sigma_0 - \sigma_{\mathrm{final}})\,\big[1 + \cos(\pi t/(T-1))\big]$.


4. Step decay. The blur is multiplied by a factor $\gamma$ every $M$ epochs: $\sigma_t = \sigma_0\,\gamma^{\lfloor t/M \rfloor}$.


5. Polynomial decay. A power‑law schedule $\sigma_t = (\sigma_0 - \sigma_{\mathrm{final}})\,(1 - t/(T-1))^p + \sigma_{\mathrm{final}}$ with polynomial power $p$.


6. Constant. Keeps $\sigma$ fixed throughout training.
Regardless of the schedule, a minimum blur value $\sigma_{\min}$ is enforced to prevent numerical underflow. In practice we choose $\sigma_0=10$, $\sigma_{\mathrm{final}}=1$ and an exponential decay rate of $0.6$ over six epochs, but other schedules can be configured via the experiment configuration.

---------------------------


Training Procedure
==================

Each experiment follows a similar training loop:

1. Data preparation. The labelled source data are split into training and validation subsets using a stratified split that preserves class proportions. When include_rotations=true the source dataset contains eight augmented views per galaxy (four rotations × two flips), otherwise only the original images are used. The target domain is unlabeled and is used only for domain adaptation and evaluation.

2. Warm‑up phase. For the first warmup_epochs epochs the model is trained exclusively with the classification loss on the source domain. The domain‑alignment loss is not computed during this phase, and early‑stopping checks are disabled to avoid stopping before adaptation begins.

3. Joint training. After warm‑up, each training batch consists of a tuple of source and target mini‑batches. We compute the forward pass on the concatenated images to obtain logits and latent features. The cross‑entropy (or focal) loss is evaluated on the source predictions, and the chosen domain‑alignment loss is evaluated on the latent features. Depending on the experiment the total loss is either $\mathcal{L}_{\text{CE}}$, $\mathcal{L}_{\text{CE}} + \lambda_{\mathrm{DA}}\mathcal{L}_{\text{DA}}$, or the trainable weighted combination described above. For adversarial training, a domain classifier is appended and its binary cross‑entropy loss is added via gradient reversal.

4. Optimisation. We use the AdamW or SGD optimiser with weight decay and optional gradient clipping (max_norm) to update the network parameters. In the trainable‑weights variant the loss weights $\eta_1$ and $\eta_2$ are learned jointly with the model and clamped after each backward pass to remain positive. In adversarial training a separate optimiser parameter group is registered for the domain classifier.

5. Evaluation and early stopping. After each epoch the network is evaluated on the target domain using accuracy, macro‑F1, AUC and other metrics. If a validation split of the source data is provided, the early‑stopping criterion can be based on the target F1/accuracy or on the training loss (configurable via early_stopping_metric). If the metric does not improve for early_stopping_patience epochs, training stops and the best model (lowest training loss) is restored.


This procedure is implemented in the BaseTrainer class and its subclasses (NoDATrainer, DAFixedLambdaTrainer, DATrainableWeightsTrainer, DATrainableWeightsSigmaTrainer and DAAdversarialTrainer). These classes override the compute_total_loss and hook functions to insert the appropriate domain‑adaptation losses, sigma schedules and gradient reversal behaviour.



Domain Adaptation Variants
==========================


We explore several domain‑adaptation strategies, each corresponding to a different trainer class and configuration. All methods share the same encoder architecture and classification loss, but differ in how (and if) they align the source and target feature distributions:


1. No domain adaptation (NoDA) — baseline. The NoDATrainer minimises only the supervised classification loss on the source domain. No target examples are used for learning, and the network is evaluated directly on the target domain. This variant provides a lower‑bound on performance and highlights the extent of covariate shift.

2. Fixed‑lambda alignment — DAFixedLambdaTrainer. In this setting a domain‑alignment loss $\mathcal{L}_{\text{DA}}$ (Sinkhorn, energy or Gaussian MMD) is added to the classification loss with a constant scaling coefficient $\lambda$:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\mathrm{DA}}\, \mathcal{L}_{\text{DA}}.$$ The hyper‑parameters method (e.g., sinkhorn) and lambda_da control the type and strength of the alignment. This variant assumes the relative importance of classification and alignment is fixed across epochs.

3. Trainable‑weights alignment — DATrainableWeightsTrainer. Here the weighting of the classification and domain‑alignment losses is learned automatically using homoscedastic uncertainty. Two parameters $(\eta_1, \eta_2)$ are initialised (e.g., $\eta_1=0.1,\,\eta_2=1.0$) and optimised jointly with the network to balance the tasks. The total loss follows the form described in the “Losses” section. After each step, the parameters are clamped to remain positive and to keep the DA weight at least a fraction of the CE weight. This variant adapts the emphasis on domain alignment based on the difficulty of the tasks.

4. Trainable weights with sigma scheduling — DATrainableWeightsSigmaTrainer. This method builds upon the trainable‑weights variant but additionally anneals the blur parameter $\sigma$ in the Sinkhorn loss. Early in training the blur is large and the loss behaves like an MMD, providing smooth gradients. As training progresses $\sigma$ is reduced following one of the schedules described in the “Sigma Scheduling” section, gradually enforcing a sharper optimal‑transport alignment. This variant performed well when combined with entropic optimal transport.

5. Adversarial domain adaptation (DANN) — DAAdversarialTrainer. Inspired by the domain‑adversarial neural network of Ganin et al. (2016), this approach replaces the explicit divergence with a domain classifier trained to distinguish source from target features. The feature extractor is encouraged to produce domain‑invariant representations by a gradient‑reversal layer that multiplies the gradient by $-\lambda_{\mathrm{grl}}$. The total loss is $$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{adv}}, $$ where $\lambda$ controls the strength of the adversarial signal (e.g., 0.25). Hyper‑parameters of the domain classifier include the latent dimension (matching the encoder output), hidden dimension and optional projection layer. This variant tends to produce smoother feature alignments but may be harder to train.

By organising the experiments according to these variants we are able to systematically compare how different alignment mechanisms and weighting strategies impact performance under covariate shift. Detailed configuration files for each combination of backbone and variant can be found in the configs/ directory.