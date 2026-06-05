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

The second analysis applies PCA and MOFA to a real multi-omics breast cancer dataset. We used the **TCGA-BRCA** cohort downloaded from [LinkedOmics](https://www.linkedomics.org/data_download/TCGA-BRCA/). Three molecular views were analyzed: RNA-seq, DNA methylation and miRNA expression. The downstream task was classification of **PAM50 molecular subtype**.

#### Methodology

The original LinkedOmics files store samples in columns and features in rows, so all matrices were transposed before analysis. PAM50 subtype was selected as the target variable. Samples without PAM50 annotation were removed, and only samples present in all three omics views were retained. This resulted in **430 common samples**.

The initial feature spaces contained:

* RNA-seq: **20 155 genes**
* DNA methylation: **20 106 features**
* miRNA: **823 features**

Missing values were almost absent, except for a small fraction in the methylation matrix. Remaining missing values were imputed using feature-wise medians. To reduce dimensionality and computational cost, the most variable features were selected independently for each view:

* RNA-seq: top **2000** most variable features
* DNA methylation: top **2000** most variable features
* miRNA: top **500** most variable features

Each view was standardized independently before fitting PCA and MOFA. This prevents one omics layer from dominating the latent representation only because of a larger numeric scale.

For each view, we fitted both PCA and MOFA with the same number of latent dimensions, `K = 5`. In addition, two multi-view baselines were evaluated:

* **PCA_concatenated** - PCA fitted on the horizontally concatenated multi-omics matrix.
* **MOFA_shared** - MOFA fitted jointly on RNA-seq, methylation and miRNA.

All embeddings were evaluated using the same downstream strategy as in Task 1: a fixed 75/25 train-test split with `seed = 1946`. A logistic regression classifier was trained on each embedding. Because PAM50 classes were imbalanced, the main metrics were **balanced accuracy** and **macro F1**.

#### Conclusions

The best result was obtained by **MOFA fitted only on RNA-seq**, with balanced accuracy of **0.865** and macro F1 of **0.834**. The second-best model was **PCA fitted only on RNA-seq**, with balanced accuracy of **0.851** and macro F1 of **0.809**. This shows that RNA-seq contained the strongest PAM50 classification signal among the tested omics views.

The multi-view models did not outperform the strongest RNA-seq-only models. **PCA_concatenated** reached balanced accuracy of **0.802**, while **MOFA_shared** reached **0.793**. Methylation contained useful but weaker subtype-related signal, with balanced accuracy around **0.75**. miRNA was the weakest view, with balanced accuracy around **0.67**.

These results are biologically expected, because PAM50 subtypes are primarily defined by gene expression profiles. Therefore, RNA-seq should naturally be the most informative view for this classification task. The confusion matrix for the best model, **MOFA_RNAseq**, showed good separation of Basal and LumA samples, while most errors occurred between Her2 and Luminal subtypes.

Overall, the empirical analysis supports the main conclusion from the simulation: **multi-view integration is not automatically better than single-view analysis**. Integration is useful when complementary information is truly distributed across views. In the TCGA-BRCA dataset, however, most PAM50-relevant signal was already captured by RNA-seq, so adding methylation and miRNA did not provide a clear predictive advantage.

---



<!-- ## Structure 

#TODO - to później uzupełnie - wszystkie funkcje dam do src/... po prostu, sam ten notebook zrobie w drugiej wersji (później ale to na czerwiec) -->

## Authors
* **Author 1:** [Max Stróżyk](https://github.com/maxi7524) – Conception of simulation matrices, execution of the synthetic non-linear manifold pipelines, and downstream evaluation architecture.
* **Author 2:** [Norbert Szala](https://github.com/NorbertSzala) – Task2: Preparation and preprocessing of the TCGA-BRCA multi-omics dataset, implementation of PCA and MOFA models for single-view and multi-view real-data analysis, downstream PAM50 subtype classification, result visualization, and biological interpretation.



