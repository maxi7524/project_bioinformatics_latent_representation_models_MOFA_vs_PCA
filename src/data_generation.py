import numpy as np

def generate_synthetic_data(n_samples=1000, seed=1946, d1=4, d2=6, d3=7):
    """
    Generate synthetic datasets for multi-view analysis scenarios A and B.

    :param n_samples: Number of samples to generate, defaults to 1000.
    :type n_samples: int, optional
    :param seed: Random state seed for reproducibility, defaults to 1946.
    :type seed: int, optional
    :param d1: Feature dimension for Modality 1 (must be >= 3), defaults to 4.
    :type d1: int, optional
    :param d2: Feature dimension for Modality 2 (must be >= 3), defaults to 6.
    :type d2: int, optional
    :param d3: Feature dimension for Modality 3 (must be >= 2), defaults to 7.
    :type d3: int, optional
    :return: A tuple containing Z space and matrices for Scenario A and Scenario B (both continuous and discrete).
    :rtype: tuple
    """
    rng = np.random.default_rng(seed)
    
    # shared latent space base
    Z = rng.uniform(-2, 2, size=(n_samples, 2))

    # scenario A
    ## case: continuous dataset
    ### labels
    heart_functional = (Z[:, 0]**2 + Z[:, 1]**2 - 1)**3 - Z[:, 0]**2 * Z[:, 1]**3
    y_a_cont = np.tanh(heart_functional) + rng.normal(0, 0.05, size=n_samples)

    ### modalities
    #### modality 1 (lower part, non-linear trend)
    mask_low = (Z[:, 1] < 0).astype(float)
    mod1_a_cont = np.zeros((n_samples, d1))
    mod1_a_cont[:, 0] = Z[:, 0] * mask_low
    mod1_a_cont[:, 1] = (Z[:, 0]**2 - 1.0) * mask_low
    mod1_a_cont[:, 2] = np.sin(Z[:, 0] * 2) * mask_low
    mod1_a_cont += rng.normal(0, 0.1, size=(n_samples, d1))

    #### modality 2 (upper part, non-linear)
    mask_high = (Z[:, 1] >= 0).astype(float)
    mod2_a_cont = np.zeros((n_samples, d2))
    mod2_a_cont[:, 0] = (Z[:, 0]**2) * mask_high
    mod2_a_cont[:, 1] = (Z[:, 1]**2) * mask_high
    mod2_a_cont[:, 2] = (Z[:, 0] * Z[:, 1]) * mask_high
    mod2_a_cont += rng.normal(0, 0.2, size=(n_samples, d2))

    #### modality 3 (pure noise)
    mod3_a_cont = rng.normal(0, 1.0, size=(n_samples, d3))

    ## case: discrete dataset
    ### labels
    radius_sq = Z[:, 0]**2 + Z[:, 1]**2
    y_a_disc = np.zeros(n_samples, dtype=int)
    y_a_disc[(radius_sq > 0.5) & (Z[:, 1] > 0)] = 1
    y_a_disc[(radius_sq > 0.5) & (Z[:, 1] <= 0)] = 2

    ### modalities
    #### modality 1 (circle and top moon, parabolic dome structure)
    mask_c0_c1 = ((y_a_disc == 0) | (y_a_disc == 1)).astype(float)
    mod1_a_disc = np.zeros((n_samples, d1))
    mod1_a_disc[:, 0] = Z[:, 0] * mask_c0_c1
    mod1_a_disc[:, 1] = Z[:, 1] * mask_c0_c1
    mod1_a_disc[:, 2] = (2.5 - (Z[:, 0]**2 + Z[:, 1]**2)) * mask_c0_c1
    mod1_a_disc += rng.normal(0, 0.1, size=(n_samples, d1))

    #### modality 2 (both moons, non-linear polar-like)
    mask_moons = ((y_a_disc == 1) | (y_a_disc == 2)).astype(float)
    r = np.sqrt(Z[:, 0]**2 + Z[:, 1]**2)
    theta = np.arctan2(Z[:, 1], Z[:, 0])
    mod2_a_disc = np.zeros((n_samples, d2))
    mod2_a_disc[:, 0] = r * mask_moons
    mod2_a_disc[:, 1] = np.sin(theta) * mask_moons
    mod2_a_disc[:, 2] = np.cos(theta) * mask_moons
    mod2_a_disc += rng.normal(0, 0.1, size=(n_samples, d2))

    #### modality 3 (noise with class-dependent centers)
    centers = {0: [-1.0, -1.0], 1: [0.0, 1.5], 2: [1.0, -1.0]}
    mod3_a_disc = np.zeros((n_samples, d3))
    for i in range(n_samples):
        mod3_a_disc[i, :2] = rng.normal(centers[y_a_disc[i]], 0.8)
    mod3_a_disc[:, 2:] = rng.normal(0, 1.2, size=(n_samples, d3 - 2))

    # scenario B
    ## case: continuous dataset
    ### labels
    y_b_cont = np.tanh(heart_functional) + rng.normal(0, 0.05, size=n_samples)

    ### modalities
    #### modality 1 (signal window, full Z mapping)
    mod1_b_cont = np.zeros((n_samples, d1))
    mod1_b_cont[:, 0] = Z[:, 0]
    mod1_b_cont[:, 1] = Z[:, 1]
    mod1_b_cont[:, 2] = (Z[:, 0]**2 + Z[:, 1]**2)
    mod1_b_cont += rng.normal(0, 0.1, size=(n_samples, d1))

    #### modality 2 (strong independent structural noise)
    Z_noise_b_cont = rng.normal(0, 1.0, size=(n_samples, 2))
    mod2_b_cont = np.zeros((n_samples, d2))
    for j in range(4):
        mod2_b_cont[:, j] = Z_noise_b_cont[:, 0] * (j + 1) + Z_noise_b_cont[:, 1] * (4 - j)
    mod2_b_cont += rng.normal(0, 0.2, size=(n_samples, d2))

    #### modality 3 (pure noise)
    mod3_b_cont = rng.normal(0, 1.5, size=(n_samples, d3))

    ## case: discrete dataset
    ### labels
    y_b_disc = y_a_disc.copy()

    ### modalities
    #### modality 1 (signal window, full Z mapping with a sharp cubic separation cascade)
    mod1_b_disc = np.zeros((n_samples, d1))
    mod1_b_disc[:, 0] = Z[:, 0]
    mod1_b_disc[:, 1] = Z[:, 1]
    r_sq = Z[:, 0]**2 + Z[:, 1]**2
    mod1_b_disc[:, 2] = (Z[:, 1]**3) + 2.0 * (r_sq > 0.5).astype(float) * np.sign(Z[:, 1])
    mod1_b_disc += rng.normal(0, 0.05, size=(n_samples, d1))

    #### modality 2 (strong independent structural noise)
    Z_noise_b_disc = rng.integers(0, 4, size=n_samples)
    mod2_b_disc = rng.normal(0, 0.3, size=(n_samples, d2))
    mod2_b_disc[:, 0] += Z_noise_b_disc * 2.0
    mod2_b_disc[:, 1] -= Z_noise_b_disc * 2.0

    #### modality 3 (pure noise)
    mod3_b_disc = rng.normal(0, 1.5, size=(n_samples, d3))

    return (Z, 
            (mod1_a_cont, mod2_a_cont, mod3_a_cont, y_a_cont),
            (mod1_a_disc, mod2_a_disc, mod3_a_disc, y_a_disc),
            (mod1_b_cont, mod2_b_cont, mod3_b_cont, y_b_cont),
            (mod1_b_disc, mod2_b_disc, mod3_b_disc, y_b_disc))