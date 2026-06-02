import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_layout_2d(Z, mod1, mod2, mod3, y, scenario_title, is_discrete=False, dot_size=9, dot_transparency=0.7):
    """
    Generate a 4-panel 2D scatter layout showing Ground Truth and Feature 0 across views.

    :param Z: Latent space coordinates matrix of shape (n_samples, 2).
    :type Z: numpy.ndarray
    :param mod1: Modality 1 feature matrix.
    :type mod1: numpy.ndarray
    :param mod2: Modality 2 feature matrix.
    :type mod2: numpy.ndarray
    :param mod3: Modality 3 feature matrix.
    :type mod3: numpy.ndarray
    :param y: Target array (continuous values or discrete class labels).
    :type y: numpy.ndarray
    :param scenario_title: Main title of the figure layout.
    :type scenario_title: str
    :param is_discrete: Flag indicating whether the target is discrete, defaults to False.
    :type is_discrete: bool, optional
    :param dot_size: Size of the markers in points, defaults to 9.
    :type dot_size: int, optional
    :param dot_transparency: Opacity alpha blending value, defaults to 0.7.
    :type dot_transparency: float, optional
    :return: None
    """
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(scenario_title, fontsize=14, fontweight='bold')

    cmap_gt = 'Set1' if is_discrete else 'coolwarm'
    ticks_gt = [0, 1, 2] if is_discrete else None

    sc0 = axes[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap_gt, edgecolor='k', alpha=dot_transparency, s=dot_size)
    axes[0].set_title("Ground Truth\nLatent Space Z")
    fig.colorbar(sc0, ax=axes[0], ticks=ticks_gt)

    axes[1].scatter(Z[:, 0], Z[:, 1], c=mod1[:, 0], cmap='viridis' if not is_discrete else 'bwr', edgecolor='k', alpha=dot_transparency, s=dot_size)
    axes[1].set_title("Modality 1: Feature 0")

    axes[2].scatter(Z[:, 0], Z[:, 1], c=mod2[:, 0], cmap='plasma' if not is_discrete else 'gnuplot', edgecolor='k', alpha=dot_transparency, s=dot_size)
    axes[2].set_title("Modality 2: Feature 0")

    if is_discrete and scenario_title.startswith("Scenario A"):
        axes[3].scatter(mod3[:, 0], mod3[:, 1], c=y, cmap='Set1', edgecolor='k', alpha=dot_transparency, s=dot_size)
        axes[3].set_title("Modality 3: Features 0 vs 1\n(Shifted Center Noise)")
    elif is_discrete and scenario_title.startswith("Scenario B"):
        axes[3].scatter(mod3[:, 0], mod3[:, 1], c=y, cmap='Set1', edgecolor='k', alpha=dot_transparency, s=dot_size)
        axes[3].set_title("Modality 2: Features 0 vs 1\n(4 Independent Clusters)")
    else:
        axes[3].scatter(Z[:, 0], Z[:, 1], c=mod3[:, 0], cmap='Greys', edgecolor='k', alpha=dot_transparency, s=dot_size)
        axes[3].set_title("Modality 3: Feature 0\n(Pure Unstructured Noise)")

    plt.tight_layout()
    plt.show()

def plot_manifold_3d(Z, mod1, mod2, mod3, y, scenario_title, is_discrete=False, dot_size=9, dot_transparency=0.7):
    """
    Generate a 4-panel 3D/2D manifold representation plot for a given scenario.

    :param Z: Latent space coordinates matrix of shape (n_samples, 2).
    :type Z: numpy.ndarray
    :param mod1: Modality 1 feature matrix.
    :type mod1: numpy.ndarray
    :param mod2: Modality 2 feature matrix.
    :type mod2: numpy.ndarray
    :param mod3: Modality 3 feature matrix.
    :type mod3: numpy.ndarray
    :param y: Target array (continuous values or discrete class labels).
    :type y: numpy.ndarray
    :param scenario_title: Main title of the figure layout.
    :type scenario_title: str
    :param is_discrete: Flag indicating whether the target is discrete, defaults to False.
    :type is_discrete: bool, optional
    :param dot_size: Size of the markers in points, defaults to 9.
    :type dot_size: int, optional
    :param dot_transparency: Opacity alpha blending value, defaults to 0.7.
    :type dot_transparency: float, optional
    :return: None
    """
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'figure.titlesize': 13})
    discrete_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    discrete_cmap = ListedColormap(discrete_colors)
    cmap_to_use = discrete_cmap if is_discrete else 'coolwarm'

    fig = plt.figure(figsize=(22, 5))
    plt.suptitle(scenario_title, fontweight='bold', y=0.98)

    # Panel 1: 2D Latent Space
    ax0 = fig.add_subplot(1, 4, 1)
    sc0 = ax0.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap_to_use, edgecolor='none', alpha=dot_transparency, s=dot_size)
    ax0.set_title("Ground Truth\nLatent Space Z (2D)")
    ax0.set_xlabel("Z1")
    ax0.set_ylabel("Z2")
    ax0.set_aspect('equal')
    if is_discrete:
        handles, _ = sc0.legend_elements()
        ax0.legend(handles, ['Circle (0)', 'Top Moon (1)', 'Bottom Moon (2)'], title="Classes", loc='best')
    else:
        fig.colorbar(sc0, ax=ax0, label="y value", shrink=0.6)

    # Panel 2: Modality 1 (3D)
    ax1 = fig.add_subplot(1, 4, 2, projection='3d')
    sc1 = ax1.scatter(mod1[:, 0], mod1[:, 1], mod1[:, 2], c=y, cmap=cmap_to_use, edgecolor='none', alpha=dot_transparency, s=dot_size)
    ax1.set_title("Modality 1: Features 0, 1, 2")
    ax1.set_xlabel("Feat 0")
    ax1.set_ylabel("Feat 1")
    ax1.set_zlabel("Feat 2")

    # Panel 3: Modality 2 (3D)
    ax2 = fig.add_subplot(1, 4, 3, projection='3d')
    sc2 = ax2.scatter(mod2[:, 0], mod2[:, 1], mod2[:, 2], c=y, cmap=cmap_to_use, edgecolor='none', alpha=dot_transparency, s=dot_size)
    ax2.set_title("Modality 2: Features 0, 1, 2")
    ax2.set_xlabel("Feat 0")
    ax2.set_ylabel("Feat 1")
    ax2.set_zlabel("Feat 2")

    # Panel 4: Modality 3 (3D or 2D depending on the scenario type)
    ax3 = fig.add_subplot(1, 4, 4, projection='3d')
    sc3 = ax3.scatter(mod3[:, 0], mod3[:, 1], mod3[:, 2], c=y, cmap=cmap_to_use, edgecolor='none', alpha=dot_transparency, s=dot_size)
    ax3.set_title("Modality 3: Features 0, 1, 2")
    ax3.set_xlabel("Feat 0")
    ax3.set_ylabel("Feat 1")
    ax3.set_zlabel("Feat 2")
    if is_discrete:
        handles, _ = sc3.legend_elements()
        ax3.legend(handles, ['Circle (0)', 'Top Moon (1)', 'Bottom Moon (2)'], title="Classes", loc='best')

    plt.tight_layout()
    plt.show()