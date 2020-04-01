# Utils
import datetime
import logging
import time
import types
import numpy as np


def apk(query, predicted, k='full'):
    """
    Computes the average precision@k.

    Parameters
    ----------
    query : int
        Query label.
    predicted : List(int)
        Ordered list where each element is a label.
    k : str or int
        If int, cutoff for retrieval is set to K
        If str, 'full' means cutoff is til the end of predicted
                'auto' means cutoff is set to number of relevant queries.

        Example:
            query = 0
            predicted = [0, 0, 1, 1, 0]
            if k == 'full', then k is set to 5
            if k == 'auto', then k is set to num of 'query' values in 'predicted',
            i.e., k=3 as there as 3 of them in 'predicted'

    Returns
    -------
    float
        Average Precision@k

    """
    assert (len(predicted) > 0)

    # Count the number of relevant items that could be retrieved
    num_hits = np.sum(predicted == query)
    if num_hits == 0:
        return 0

    # Resolve k in case is not a number
    if k == 'auto':
        k = num_hits
    elif k == 'full':
        k = len(predicted)
    else:
        assert isinstance(k, int)
    assert (k > 0) and (k <= len(predicted))

    # Truncate the list to the number of desired elements which gets taken into account
    predicted = np.array(predicted[:k])

    # Non-vectorized version.
    # score = 0.0  # The score is the precision@i integrated over i=1:k
    # num_hits = 0.0
    #
    # for i, p in enumerate(predicted):
    #     if p == query:
    #         num_hits += 1.0
    #         score += num_hits / (i + 1.0)
    #
    # return score / k_or_num_hits

    # Make an empty array for relevant queries.
    relevant = np.zeros(len(predicted))

    # Find all locations where the predicted value matches the query and vice-versa.
    hit_locs = (predicted == query)
    non_hit_locs = np.logical_not(hit_locs)

    # Set all `hit_locs` to be 1. [0,0,0,0,0,0] -> [0,1,0,1,0,1]
    relevant[hit_locs] = 1
    # Compute the sum of all elements till the particular element. [0,1,0,1,0,1] -> [0,1,1,2,2,3]
    relevant = np.cumsum(relevant)
    #  Set all `non_hit_locs` to be zero. [0,1,1,2,2,3] -> [0,1,0,2,0,3]
    relevant[non_hit_locs] = 0
    # Divide element-wise by [0/1,1/2,0/3,2/4,0/5,3/6] and sum the array.
    score = np.sum(np.divide(relevant, np.arange(1, relevant.shape[0] + 1)))

    return score / min(k, num_hits)


def mapk(query, predicted, k=None, workers=1):
    """Compute the mean Average Precision@K.

    Parameters
    ----------
    query : list
        List of queries.
    predicted : list of list, or generator to list of lists
        Predicted responses for each query. Supports chunking with slices in
        the first dimension.
    k : str or int
        If int, cutoff for retrieval is set to `k`
        If str, 'full' means cutoff is til the end of predicted
                'auto' means cutoff is set to number of relevant queries.
        For e.g.,
            `query` = 0
            `predicted` = [0, 0, 1, 1, 0]
            if `k` == 'full', then `k` is set to 5
            if `k` == 'auto', then `k` is set to num of `query` values in `predicted`,
            i.e., `k`=3 as there as 3 of them in `predicted`.
    workers : int
        Number of parallel workers used to compute the AP@k

    Returns
    -------
    float
        The mean average precision@K.
    dict{label, float}
        The per class mean averages precision @k
    """

    # If distances come from pairwise_distances_chunked they must be flattened
    # since apk operates on a per-row basis.
    if type(predicted) is types.GeneratorType:
        predicted = [row for nested in predicted for row in nested]

    results = np.array([apk(q, p, k) for q, p in zip(query, predicted)])
    per_class_mapk = {str(l): np.mean(np.array(results)[np.where(query == l)[0]]) for l in np.unique(query)}
    return np.mean(results), per_class_mapk
    # The overhead of the pool is killing any possible speedup.
    # In order to make this parallel (if ever needed) one should create a Process class which swallows
    # 1/`workers` part of `vals`, such that only `workers` threads are created.

    # if workers == 1:
    #     return np.mean([_apk(q, p, k) for q, p in zip(query, predicted)])
    # with Pool(workers) as pool:
    #     vals = [[q, p, k] for q, p in zip(query, predicted)]
    #     aps = pool.starmap(_apk, vals)
    #     return np.mean(aps)


def compute_mapk(distances, labels, k, workers=None):
    """Convenience function to convert a grid of pairwise distances to predicted
    elements, to evaluate mean average precision (at K).

    Parameters
    ----------
    distances : ndarray
        A numpy array containing pairwise distances between all elements
    labels : list
        Ground truth labels for every element
    k : int
        Maximum number of predicted elements

    Returns
    -------
    float
        The mean average precision@K.
    dict{label, float}
        The per class mean averages precision @k
    """

    def chunked_sorting(distances):
        '''Sorts a _chunked_ pairwise distance matrix.

            Parameters
            ----------
            distances : generator of ndarray
                A generator yielding numpy arrays containing pairwise
                distances between a subset of all elements. Suitable for
                combination with sklearn.metrics.pairwise_distances_chunked
                which slices the matrix along the first dimenstion (i.e. one
                can iterate over entire rows easily).

            Returns
            -------
            A generator of sorted chunks of the input matrix.
        '''
        for i, chunk in enumerate(distances):
            # Fetch the index of the lowest `max_count` (k) elements
            if k != 'full':
                ind = np.argpartition(chunk, max_count - 1)[:, :max_count]
                # Find the sorting sequence according to the shortest chunk selected from `ind`
                ssd = np.argsort(np.array(chunk)[np.arange(chunk.shape[0])[:, None], ind], axis=1)
                # Consequently sort `ind`
                ind = ind[np.arange(ind.shape[0])[:, None], ssd]
                # Now `ind` contains the sorted indexes of the lowest `max_count` (k) elements
            else:
                # If we're in full mode, just to the sorting directly
                ind = np.argsort(chunk)

            # Resolve the labels of the elements referred by `ind`
            # sorted_predictions = [list(labels[row][1:]) for row in ind]
            sorted_predictions = np.empty(shape=(chunk.shape[0],
                                                 chunk.shape[1]-1),
                                          dtype=np.int)
            for i, row in enumerate(ind):
                sorted_predictions[i, :] = labels[row][1:]

            yield sorted_predictions

    # In case of non-chunked input we wrap to ensure uniform treatment
    if type(distances) is not types.GeneratorType:
        distances = [distances]

    # Resolve k
    k = k if k == 'auto' or k == 'full' else int(k)

    # Reduce the size of distances that would anyway not be used afterwards. This makes sorting them faster.
    max_count = k
    if k == 'full':
        max_count = len(labels)
    if k == 'auto':
        # Take the highest frequency in the labels i.e. the highest possible 'auto' value for all entries
        max_count = np.max(np.unique(labels, return_counts=True)[1])

    # Do lazy sorting of the distance matrix
    sorted_predictions_chunked = chunked_sorting(distances)

    if workers is None:
        workers = 16 if k == 'auto' or k == 'full' else 1

    return mapk(labels, sorted_predictions_chunked, k, workers)
