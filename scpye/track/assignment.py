import numpy as np
from scipy.optimize import linear_sum_assignment as skla


def pad_cost_matrix(cost_matrix, unassigned_cost):
    """
    Pad cost matrix to handle unassignment
    :param cost_matrix: M x N matrix, M - tracks, N - detections
    :param unassigned_cost:  cost of unassignment
    :return: padded cost matrix
    """

    # TODO: the unassigned_cost here is like a threshold value, when the cost is larger than this, the two will not be assigned
    if unassigned_cost == 0.0:
        return cost_matrix

    r, c = np.shape(cost_matrix)
    # padded size
    n = r + c
    float_max = np.finfo(np.float).max
    padded_cost_matrix = np.ones((n, n)) * float_max

    # fill the padded cost matrix
    padded_cost_matrix[:r, :c] = cost_matrix
    padded_cost_matrix[r:, c:] = 0.0
    padded_cost_matrix[range(r), range(c, n)] = unassigned_cost
    padded_cost_matrix[range(r, n), range(c)] = unassigned_cost
    return padded_cost_matrix


def hungarian_assignment(cost_matrix, unassigned_cost=1.2):
    """
    This is the equivalent of matlab's assignDetectionsToTracks
    :param cost_matrix: M x N matrix, M - tracks, N - detections
    :param unassigned_cost: cost of unassignment
    :return: matches, unassigned tracks, unassigned detections
    """
    padded_cost_matrix = pad_cost_matrix(cost_matrix, unassigned_cost)
    # TODO: refer to here https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/linear_assignment_.py
    #This is outdated, updating to linearn_sum_assignment and converting to numpy array
    assignment = skla(padded_cost_matrix)
    assignment = np.asarray(assignment)
    assignment = np.transpose(assignment)

    # Figure out matches, unassigned bboxes1 and unassigned bboxes2
    n1, n2 = np.shape(cost_matrix)
    ind1 = assignment[:, 0]
    ind2 = assignment[:, 1]

    # Correct matches
    match_inds = (ind1 < n1) & (ind2 < n2)
    matches = assignment[match_inds]
    matches = np.atleast_2d(matches)

    # Unassigned bboxes1 and bboxes2
    un_ind1 = (ind1 < n1) & (ind2 >= n2)
    un_ind2 = (ind1 >= n1) & (ind2 < n2)

    unassigned1 = ind1[un_ind1]
    unassigned2 = ind2[un_ind2]

    return matches, unassigned1, unassigned2
