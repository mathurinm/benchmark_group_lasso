def convert_to_group_index(groups, n_features):
    """
    Assign features to groups for Group Lasso.

    Parameters:
    - groups: Either an int, a list of ints, or a list of lists of ints
      * int: Number of groups
      * list of ints: Number of features in each group
      * list of lists of ints: Each sublist represents a group of features
    - n_features: Total number of features

    Returns:
    - List[int]: i-th element represents the group of the i-th feature
    """
    group_assignment = [-1] * n_features  # Initialize with -1 for unassigned

    if isinstance(groups, int):
        # Case 1: Single integer input
        num_groups = groups
        features_per_group = n_features // num_groups

        # Distribute features among groups evenly, if possible
        for feature_index in range(n_features):
            group_assignment[
                feature_index] = feature_index // features_per_group

    elif isinstance(groups, list) and all(isinstance(g, int) for g in groups):
        # Case 2: List of integers input
        feature_index = 0
        for group_index, num_features_in_group in enumerate(groups):
            for _ in range(num_features_in_group):
                if feature_index < n_features:
                    group_assignment[feature_index] = group_index
                    feature_index += 1
                else:
                    break

    elif isinstance(groups, list) and all(isinstance(g, list) for g in groups):
        # Case 3: List of lists input
        for group_index, group in enumerate(groups):
            for feature in group:
                if 0 <= feature < n_features:
                    group_assignment[feature] = group_index

    else:
        raise ValueError("Invalid input format for groups")

    return group_assignment
