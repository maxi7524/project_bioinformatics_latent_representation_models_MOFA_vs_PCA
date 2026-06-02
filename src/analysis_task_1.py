import mofapy2
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# Global parameters matching the main script configuration
seed = 1946
test_size = 0.25
n_comp_pca = 2
n_factors_mofa = 3
dot_transparency = 0.7
dot_size = 9

discrete_colors = ['#e41a1c', '#377eb8', '#4daf4a']
discrete_cmap = ListedColormap(discrete_colors)


def evaluate_embeddings_rf(X_embeddings, y, is_discrete):
    """
    Evaluate structural embedding performance via Logistic Regression or Random Forest.

    :param X_embeddings: Latent embedding matrix.
    :type X_embeddings: numpy.ndarray
    :param y: Target array.
    :type y: numpy.ndarray
    :param is_discrete: Flag indicating continuous vs discrete mode.
    :type is_discrete: bool
    :return: Computed evaluation score.
    :rtype: float
    """
    if is_discrete:
        X_train, X_test, y_train, y_test = train_test_split(
            X_embeddings, y, test_size=test_size, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return accuracy_score(y_test, preds)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_embeddings, y, test_size=test_size, random_state=seed
        )
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        return r2_score(y_test, preds)


def evaluate_embeddings_kmeans(X_embeddings, y, is_discrete):
    """
    Evaluate structural embedding performance using unsupervised KMeans clustering as a base.

    :param X_embeddings: Latent embedding matrix.
    :type X_embeddings: numpy.ndarray
    :param y: Target array.
    :type y: numpy.ndarray
    :param is_discrete: Flag indicating continuous vs discrete mode.
    :type is_discrete: bool
    :return: Computed evaluation score based on cluster assignments.
    :rtype: float
    """
    n_clusters = 3 if is_discrete else 4
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = km.fit_predict(X_embeddings)

    X_train, X_test, y_train, y_test = train_test_split(
        clusters.reshape(-1, 1), y, test_size=test_size, random_state=seed, stratify=y if is_discrete else None
    )

    if is_discrete:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return accuracy_score(y_test, preds)
    else:
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        return r2_score(y_test, preds)


def run_random_forest_analysis(m1, m2, m3, y, scenario_title, is_discrete):
    """
    Perform structural embedding extraction and visualization using the Random Forest workflow.

    :param m1: Modality 1 matrix.
    :param m2: Modality 2 matrix.
    :param m3: Modality 3 matrix.
    :param y: Labels array.
    :param scenario_title: Title header for console outputs and plots.
    :param is_discrete: Flag indicating whether dataset properties are discrete.
    """
    # PCA section
    pca_obj_m1 = PCA(n_components=n_comp_pca, random_state=seed)
    pca_m1 = pca_obj_m1.fit_transform(m1)
    
    pca_obj_m2 = PCA(n_components=n_comp_pca, random_state=seed)
    pca_m2 = pca_obj_m2.fit_transform(m2)
    
    pca_obj_m3 = PCA(n_components=n_comp_pca, random_state=seed)
    pca_m3 = pca_obj_m3.fit_transform(m3)
    
    concat_data = np.hstack([m1, m2, m3])
    pca_obj_concat = PCA(n_components=n_comp_pca, random_state=seed)
    pca_concat = pca_obj_concat.fit_transform(concat_data)
    
    # MOFA section with stdout redirection to suppress logging output
    mofa_kwargs = {"n_factors": n_factors_mofa, "seed": seed, "quiet": True}
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        mdata_m1 = mu.MuData({"mod1": sc.AnnData(m1)})
        mu.tl.mofa(mdata_m1, **mofa_kwargs)
        mofa_m1 = mdata_m1.obsm["X_mofa"]
        
        mdata_m2 = mu.MuData({"mod2": sc.AnnData(m2)})
        mu.tl.mofa(mdata_m2, **mofa_kwargs)
        mofa_m2 = mdata_m2.obsm["X_mofa"]
        
        mdata_m3 = mu.MuData({"mod3": sc.AnnData(m3)})
        mu.tl.mofa(mdata_m3, **mofa_kwargs)
        mofa_m3 = mdata_m3.obsm["X_mofa"]
        
        mdata_shared = mu.MuData({"mod1": sc.AnnData(m1), "mod2": sc.AnnData(m2), "mod3": sc.AnnData(m3)})
        mu.tl.mofa(mdata_shared, **mofa_kwargs)
        mofa_shared = mdata_shared.obsm["X_mofa"]
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    # Downstream evaluation mapping
    embeddings = {
        "PCA: Modality 1": pca_m1, "PCA: Modality 2": pca_m2, "PCA: Modality 3": pca_m3, "PCA: Concatenated": pca_concat,
        "MOFA: Modality 1": mofa_m1, "MOFA: Modality 2": mofa_m2, "MOFA: Modality 3": mofa_m3, "MOFA: Shared Joint": mofa_shared
    }
    
    metric_name = 'Test Accuracy' if is_discrete else 'Test Non-linear R2 (Random Forest)'
    results = {name: evaluate_embeddings_rf(emb, y, is_discrete) for name, emb in embeddings.items()}
    df_res = pd.DataFrame.from_dict(results, orient='index', columns=[metric_name])
    
    cmap_to_use = discrete_cmap if is_discrete else 'coolwarm'
    
    # Plotting row 1: PCA Embeddings (Full data, no downsampling)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    fig.suptitle(f"{scenario_title} (Random Forest Pipeline) - PCA Dimensionality Reduction (Colored by y)", fontweight='bold', fontsize=12)
    pca_list = [pca_m1, pca_m2, pca_m3, pca_concat]
    pca_objs = [pca_obj_m1, pca_obj_m2, pca_obj_m3, pca_obj_concat]
    titles_pca = ["Modality 1", "Modality 2", "Modality 3", "Concatenated views"]
    
    for idx, (emb, obj, title) in enumerate(zip(pca_list, pca_objs, titles_pca)):
        scat = axes[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap_to_use, alpha=dot_transparency, s=dot_size, edgecolor='none')
        axes[idx].set_title(title)
        axes[idx].set_xlabel(f"PC1 ({obj.explained_variance_ratio_[0]*100:.1f}%)")
        axes[idx].set_ylabel(f"PC2 ({obj.explained_variance_ratio_[1]*100:.1f}%)")
        if not is_discrete:
            fig.colorbar(scat, ax=axes[idx], shrink=0.6)
        elif idx == 3:
            handles, _ = scat.legend_elements()
            axes[idx].legend(handles, ['Circle (0)', 'Top Moon (1)', 'Bottom Moon (2)'], title="Classes", loc='best')
    plt.tight_layout()
    plt.show()

    # Plotting row 2: MOFA Embeddings (Full data, no downsampling)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    fig.suptitle(f"{scenario_title} (Random Forest Pipeline) - MOFA Factor Embeddings (Colored by y)", fontweight='bold', fontsize=12)
    mofa_list = [mofa_m1, mofa_m2, mofa_m3, mofa_shared]
    titles_mofa = ["Modality 1 Specific", "Modality 2 Specific", "Modality 3 Specific", "Shared Joint Spaces"]
    
    for idx, (emb, title) in enumerate(zip(mofa_list, titles_mofa)):
        scat = axes[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap_to_use, alpha=dot_transparency, s=dot_size, edgecolor='none')
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Factor 1")
        axes[idx].set_ylabel("Factor 2")
        if not is_discrete:
            fig.colorbar(scat, ax=axes[idx], shrink=0.6)
        elif idx == 3:
            handles, _ = scat.legend_elements()
            axes[idx].legend(handles, ['Circle (0)', 'Top Moon (1)', 'Bottom Moon (2)'], title="Classes", loc='best')
    plt.tight_layout()
    plt.show()
    
    print(f"\n--- {scenario_title} Downstream Performance Analysis Table ---")
    print(df_res.to_markdown())
    print("\n" + "="*80 + "\n")


def run_kmeans_analysis(m1, m2, m3, y, scenario_title, is_discrete):
    """
    Perform structural embedding extraction and unsupervised clustering performance mapping using KMeans convention.

    :param m1: Modality 1 matrix.
    :param m2: Modality 2 matrix.
    :param m3: Modality 3 matrix.
    :param y: Labels array.
    :param scenario_title: Title header for console outputs and plots.
    :param is_discrete: Flag indicating whether dataset properties are discrete.
    """
    # PCA section
    pca_obj_m1 = PCA(n_components=n_comp_pca, random_state=seed)
    pca_m1 = pca_obj_m1.fit_transform(m1)
    
    pca_obj_m2 = PCA(n_components=n_comp_pca, random_state=seed)
    pca_m2 = pca_obj_m2.fit_transform(m2)
    
    pca_obj_m3 = PCA(n_components=n_comp_pca, random_state=seed)
    pca_m3 = pca_obj_m3.fit_transform(m3)
    
    concat_data = np.hstack([m1, m2, m3])
    pca_obj_concat = PCA(n_components=n_comp_pca, random_state=seed)
    pca_concat = pca_obj_concat.fit_transform(concat_data)
    
    # MOFA section with stdout redirection to suppress logging output
    mofa_kwargs = {"n_factors": n_factors_mofa, "seed": seed, "quiet": True}
    
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        mdata_m1 = mu.MuData({"mod1": sc.AnnData(m1)})
        mu.tl.mofa(mdata_m1, **mofa_kwargs)
        mofa_m1 = mdata_m1.obsm["X_mofa"]
        
        mdata_m2 = mu.MuData({"mod2": sc.AnnData(m2)})
        mu.tl.mofa(mdata_m2, **mofa_kwargs)
        mofa_m2 = mdata_m2.obsm["X_mofa"]
        
        mdata_m3 = mu.MuData({"mod3": sc.AnnData(m3)})
        mu.tl.mofa(mdata_m3, **mofa_kwargs)
        mofa_m3 = mdata_m3.obsm["X_mofa"]
        
        mdata_shared = mu.MuData({"mod1": sc.AnnData(m1), "mod2": sc.AnnData(m2), "mod3": sc.AnnData(m3)})
        mu.tl.mofa(mdata_shared, **mofa_kwargs)
        mofa_shared = mdata_shared.obsm["X_mofa"]
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    # Downstream evaluation mapping (KMeans integration strategy)
    embeddings = {
        "PCA: Modality 1": pca_m1, "PCA: Modality 2": pca_m2, "PCA: Modality 3": pca_m3, "PCA: Concatenated": pca_concat,
        "MOFA: Modality 1": mofa_m1, "MOFA: Modality 2": mofa_m2, "MOFA: Modality 3": mofa_m3, "MOFA: Shared Joint": mofa_shared
    }
    
    metric_name = 'Test Accuracy (KMeans)' if is_discrete else 'Test Non-linear R2 (KMeans Integration)'
    results = {name: evaluate_embeddings_kmeans(emb, y, is_discrete) for name, emb in embeddings.items()}
    df_res = pd.DataFrame.from_dict(results, orient='index', columns=[metric_name])
    
    cmap_to_use = discrete_cmap if is_discrete else 'coolwarm'
    
    # Plotting row 1: PCA Embeddings (Full data, no downsampling)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    fig.suptitle(f"{scenario_title} (KMeans Pipeline) - PCA Dimensionality Reduction (Colored by y)", fontweight='bold', fontsize=12)
    pca_list = [pca_m1, pca_m2, pca_m3, pca_concat]
    pca_objs = [pca_obj_m1, pca_obj_m2, pca_obj_m3, pca_obj_concat]
    titles_pca = ["Modality 1", "Modality 2", "Modality 3", "Concatenated views"]
    
    for idx, (emb, obj, title) in enumerate(zip(pca_list, pca_objs, titles_pca)):
        scat = axes[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap_to_use, alpha=dot_transparency, s=dot_size, edgecolor='none')
        axes[idx].set_title(title)
        axes[idx].set_xlabel(f"PC1 ({obj.explained_variance_ratio_[0]*100:.1f}%)")
        axes[idx].set_ylabel(f"PC2 ({obj.explained_variance_ratio_[1]*100:.1f}%)")
        if not is_discrete:
            fig.colorbar(scat, ax=axes[idx], shrink=0.6)
        elif idx == 3:
            handles, _ = scat.legend_elements()
            axes[idx].legend(handles, ['Circle (0)', 'Top Moon (1)', 'Bottom Moon (2)'], title="Classes", loc='best')
    plt.tight_layout()
    plt.show()

    # Plotting row 2: MOFA Embeddings (Full data, no downsampling)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    fig.suptitle(f"{scenario_title} (KMeans Pipeline) - MOFA Factor Embeddings (Colored by y)", fontweight='bold', fontsize=12)
    mofa_list = [mofa_m1, mofa_m2, mofa_m3, mofa_shared]
    titles_mofa = ["Modality 1 Specific", "Modality 2 Specific", "Modality 3 Specific", "Shared Joint Spaces"]
    
    for idx, (emb, title) in enumerate(zip(mofa_list, titles_mofa)):
        scat = axes[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap_to_use, alpha=dot_transparency, s=dot_size, edgecolor='none')
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Factor 1")
        axes[idx].set_ylabel("Factor 2")
        if not is_discrete:
            fig.colorbar(scat, ax=axes[idx], shrink=0.6)
        elif idx == 3:
            handles, _ = scat.legend_elements()
            axes[idx].legend(handles, ['Circle (0)', 'Top Moon (1)', 'Bottom Moon (2)'], title="Classes", loc='best')
    plt.tight_layout()
    plt.show()
    
    print(f"\n--- {scenario_title} Downstream Performance Analysis Table (KMeans Integration) ---")
    print(df_res.to_markdown())
    print("\n" + "="*80 + "\n")