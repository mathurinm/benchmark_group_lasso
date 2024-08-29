def convert_to_group_index(groups, n_features):
    """
    Convert the input into a list of lists representing the group indices.

    Parameters
    ----------
    groups : int | list of ints | list of lists of ints

    n_features : int
        Total number of features.

    Returns
    -------
    group_index : list of lists of ints
    """
    if isinstance(groups, int):
        # Create groups of contiguous features of size 'groups'
        group_index = [list(range(i, i + groups)) for i in range(0, n_features,
                                                                 groups)]
    elif isinstance(groups, list) and isinstance(groups[0], int):
        # Create groups with specified sizes
        group_index = []
        start_idx = 0
        for size in groups:
            group_index.append(list(range(start_idx, start_idx + size)))
            start_idx += size
    elif isinstance(groups, list) and isinstance(groups[0], list):
        # Already a list of lists, do nothing
        group_index = groups
    else:
        raise ValueError(
            "Unsupported group format. Must be an int, list of ints, or list"
            "of lists of ints.")

    return group_index
