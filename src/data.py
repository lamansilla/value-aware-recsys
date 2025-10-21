import numpy as np


def generate_data(
    scenario="I",
    theta=0.9,
    n_customers=150,
    n_products=1500,
    max_expenditure=10000,
    noise_ratio=0.05,
    noise_discount=0.2,
    random_seed=None,
):
    """
    Generates synthetic customer expenditure data based on predefined consumer types.

    Parameters
    ----------
    scenario : str, optional (default="I")
        Data generation scenario:
        - "I": 2 customer types (A and B)
        - "II": 3 types (A, B, C)
        - "III": 4 types (A, B, C, D)

    theta : float or tuple(float, float), optional (default=0.9)
        Proportion of preferred products that each customer buys.
        If it is a float, all customers buy the same proportion.
        If it is a tuple (min, max), the proportion is uniformly sampled for each customer.

    n_customers : int, optional (default=150)
        NNumber of customers to generate for each type.
        The total will be n_customers * number of types in the scenario.

    n_products : int, optional (default=1500)
        Total number of products in the catalog.

    max_expenditure : float, optional (default=10000)
        Maximum possible expenditure per product.

    noise_ratio: float, optional (default=0.05)
        Proportion of purchases outside the preferred range (noise).

    noise_discount: float, optional (default=0.2)
        Discount factor applied to expenditure on non-preferred products.

    random_seed : int, optional (default=None)
        Seed for reproducibility. If None, no seed is set.

    Returns
    -------
    tuple (data, true_labels):
        data: numpy.ndarray of shape (n_customers * number of types, n_products) with expenditures
        true_labels: array with the true type of each customer (0=A, 1=B, etc.)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(theta, (int, float)):
        theta_min = theta_max = float(theta)
    elif isinstance(theta, tuple) and len(theta) == 2:
        theta_min, theta_max = map(float, theta)
    else:
        raise ValueError("theta must be a float or a tuple of two floats.")

    # Definition of consumer types
    type_specs = {"I": ["A", "B"], "II": ["A", "B", "C"], "III": ["A", "B", "C", "D"]}

    if scenario not in type_specs:
        raise ValueError(
            f"Scenario '{scenario}' is not valid. Choose from {list(type_specs.keys())}."
        )

    data = []
    true_labels = []

    for type_idx, t in enumerate(type_specs[scenario]):

        # Configuration of consumer types by scenario
        if t == "A":
            pref_range = (0, int(0.8 * n_products))
            alpha = 0.1
        elif t == "B":
            pref_range = (int(0.9 * n_products), n_products)
            alpha = 1.0
        elif t == "C":
            pref_range = (int(0.6 * n_products), int(0.95 * n_products))
            alpha = 0.4
        elif t == "D":
            pref_range = (int(0.4 * n_products), int(0.6 * n_products))
            alpha = 0.4

        pref_indices = np.arange(*pref_range)
        if len(pref_indices) == 0:
            continue

        # Generate data for each customer of this type
        for _ in range(n_customers):
            # Determine preferred product proportion
            theta_client = np.random.uniform(theta_min, theta_max)
            n_selected_pref = max(1, int(theta_client * len(pref_indices)))

            # Select preferred products
            selected_pref = np.random.choice(
                pref_indices, size=n_selected_pref, replace=False
            )

            # Assign expenditure to preferred products
            expenditures = np.zeros(n_products)
            expenditures[selected_pref] = (
                np.random.uniform(0, max_expenditure, n_selected_pref) * alpha
            )

            # Noise: purchases outside the preferred range
            if noise_ratio > 0:
                non_pref_indices = np.setdiff1d(np.arange(n_products), pref_indices)
                if len(non_pref_indices) > 0:
                    n_noise = max(1, int(noise_ratio * n_selected_pref))
                    selected_noise = np.random.choice(
                        non_pref_indices, size=n_noise, replace=False
                    )

                    # The expenditure on non-preferred products is lower
                    expenditures[selected_noise] = (
                        np.random.uniform(0, max_expenditure * noise_discount, n_noise)
                        * alpha
                    )

            data.append(expenditures)
            true_labels.append(type_idx)

    return np.array(data), np.array(true_labels)


def generate_masked_data(data, beta=0.9, random_seed=None):
    """
    Generates training and test sets by masking products.

    Parameters
    ----------
    data : numpy.ndarray
        Original transaction matrix of shape (n_users, n_products).

    beta : float, optional (default=0.9)
        Proportion of purchased products to be masked per user.
        Higher values imply less observed data.

    random_seed : int, optional (default=None)
        Seed for reproducibility. If None, no seed is set.

    Returns
    -------
    tuple (S_train, S_test, mask)
        S_train: Normalized training matrix (masked products = 0)
        S_test: Test matrix with original values only for masked products
        mask: Boolean matrix indicating masked positions
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_users, _ = data.shape
    mask = np.zeros_like(data, dtype=bool)

    for i in range(n_users):
        # Products purchased by the user (indices with value > 0)
        purchased_indices = np.where(data[i] > 0)[0]
        n_purchased = len(purchased_indices)

        if n_purchased > 1:
            # NNumber of products to mask
            n_masked = max(1, int(beta * n_purchased))
            n_masked = min(
                n_masked, n_purchased - 1
            )  # Leave at least one product unmasked

            # Select products to mask
            masked_indices = np.random.choice(
                purchased_indices, size=n_masked, replace=False
            )
            mask[i, masked_indices] = True

    # Training set
    S_train = data.copy()
    S_train[mask] = 0

    # Normalization
    total_revenue_train = np.sum(S_train)
    if total_revenue_train > 0:
        S_train /= total_revenue_train

    # Test set (only masked values)
    S_test = np.where(mask, data, 0)

    return S_train, S_test, mask
