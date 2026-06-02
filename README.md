# project bioinformatics latent representation models MOFA vs PCA

## Project Overview
This project focus on comparison between two laten representation models - MOFA and PCA. It was done as a project for *Modeling of complex biological systems* course on Warsaw University.

The project is structured into two separate analyses:
1. **Theoretical Simulation Framework:** comparing  representations against synthetic non-linear manifolds.
2. **Real Multi-Omics Application:** Evaluating the generalization power of these representation models on an empirical, peer-reviewed multi-omics dataset.

---

## Analysis

### Analysis 1: Synthetic Data Simulation

#### Methodology
The foundational architecture relies on a uniform two-dimensional latent space $Z \in \mathbb{R}^{2}$ bounded within $[-2, 2]^2$, with sample parameters fixed uniformly at $N = 1000$ and `seed = 1946`. We implement two main architectures across both continuous (gradient mappings evaluated using Random Forest Regressors) and discrete configurations (categorical classes mapping half-moons and internal circles evaluated using Multinomial Logistic Regression):

* **Scenario A (Distributed Signal):** The predictive signal driving the target variable $y$ is fragmented across different views via mutually exclusive orthogonal mapping masks ($\mathcal{G}_1 \cap \mathcal{G}_2 = \emptyset$). No single modality contains the complete underlying generative rule.
* **Scenario B (Concentrated Signal):** The signal is contained entirely within a single unmasked window (`Modality 1`), while adjacent views introduce structured, high-variance noise fields completely independent of the downstream target ($I(X^2; y) = 0$).

#### Conclusions
* **Distributed Signal Framework:** Single-view representations produce low predictive performance because individual sub-$\sigma$-algebras lack full geometric context. The multi-view joint strategies (`PCA: Concatenated` and `MOFA: Shared Joint`) show a slight but distinct and clear performance progression ($R^2 \approx 0.38$, $\text{Accuracy} \approx 0.99$). This behavior matches theoretical expectations; because the views share a balanced variance scale, both horizontal concatenation and factor alignment capture the shared topology with comparable efficiency.
* **Concentrated Signal Framework:** Under extreme variance imbalance, `PCA: Concatenated` fails completely in continuous tracks ($R^2 = -0.0124$). This happens because PCA maximizes global variance without considering target labels, causing the independent noise features in `Modality 2` to dominate the first two components. Probabilistic single-view extraction (`MOFA: Modality 1`) achieves the highest continuous performance ($R^2 = 0.9326$) by separating isotropic noise from the underlying manifold. In discrete configurations, however, PCA retains high accuracy ($\text{Accuracy} = 1.0000$) due to the high-variance cascade structure of the categorical boundaries, demonstrating that variance maximization aligns with downstream classification labels only under specific geometric conditions.

***

### Analysis 2: Empirical Real Multi-Omics Dataset
*#TODO - Norbert*

#### Methodology
*#TODO - Norbert*

#### Conclusions
*#TODO - Norbert*

---



<!-- ## Structure 

#TODO - to później uzupełnie - wszystkie funkcje dam do src/... po prostu, sam ten notebook zrobie w drugiej wersji (później ale to na czerwiec) -->

## Authors
* **Author 1:** [Max Stróżyk](https://github.com/maxi7524) – Conception of simulation matrices, execution of the synthetic non-linear manifold pipelines, and downstream evaluation architecture.
* **Author 2:** [Norbert Szala](https://github.com/NorbertSzala) – *#TODO - Norbert*



