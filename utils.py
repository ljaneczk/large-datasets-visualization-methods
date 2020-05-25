import random

from sklearn.neighbors import DistanceMetric
import numpy as np
from sklearn.neighbors import kneighbors_graph

dist = DistanceMetric.get_metric('euclidean')

N = 10
M = 5
gamma = 7


def count_P(X, sigma, adj_matrix):
    P_cond = np.zeros(N, N)
    for i in range(len(X)):
        # (4)
        suma = 0
        for k in range(len(X)):
            if k != i:
                suma += count_exp_dist(X, i, k, sigma)
        for j in range(len(X)):
            if j in adj_matrix[i]:
                P_cond[i, j] = probability_i_j(X, i, j, sigma, suma)
            else:
                P_cond[i, j] = 0

    # (5)
    P = np.zeros(N, N)
    for i in range(N):
        for j in range(N):
            P[i, j] = p_i_j(P_cond[i, j], P_cond[j, i], N)
    return P


def probability_i_j(X, i, j, sigma, suma):
    return count_exp_dist(X, i, j, sigma) / suma


def count_exp_dist(X, i, j, sigma):
    return np.exp(-dist(X[i], X[j]) ** 2 / (2 * sigma ** 2))


def p_i_j(pi_cond_j, pj_cond_i, N):
    return (pi_cond_j + pj_cond_i) / (2 * N)


def count_Q(Y):
    Q = np.zeros(N, N)
    for i in range(len(Y)):
        # (6)
        suma = 0
        for k in range(len(Y)):
            if k != i:
                suma += count_6(Y[i], Y[k]) ** (-1)
        for j in range(len(Y)):
            Q[i, j] = count_6(Y[i], Y[j]) ** (-1) / suma
    return Q


def count_6(Y_i, Y_j):
    return 1 + dist(Y_i, Y_j) ** 2


def count_p_i_cond_el(X, sigma, i, el, are_neighbours):
    if not are_neighbours:
        return 0
    suma = 0
    for k in range(len(X)):
        if k != i:
            suma += count_exp_dist(X, i, k, sigma)
    return probability_i_j(X, i, el, sigma, suma)


def count_p_i_el(X, sigma, i, el, are_neighbours, N):
    p_i_cond_el = count_p_i_cond_el(X, sigma, i, el, are_neighbours)
    p_el_cond_i = count_p_i_cond_el(X, sigma, el, i, are_neighbours)
    return p_i_cond_el + p_el_cond_i / 2 * N


def dv(X, vertexes, adj_matrix):
    # (11)
    original_vertexes = np.copy(vertexes)
    n = len(vertexes)
    i = random.randint(0, n)
    y_i = vertexes[i]
    el = random.randint(0, n)
    while el == i:
        el = random.randint(0, n)
    y_el = vertexes[el]  # positive sample
    rest_of_ys = np.delete(vertexes, [i, el], None)  # negative samples
    sigma = 0.1
    are_neighbours = el in adj_matrix[i]
    p_i_L = count_p_i_el(X, sigma, i, el, are_neighbours, N)
    suma = - 2 * p_i_L * (y_i - y_el) / count_6(y_i, y_el)
    for k in rest_of_ys:
        if k in adj_matrix[i]:
            y_el_k = rest_of_ys[k]
            are_neighbours = el in adj_matrix[k]
            suma += 2 * gamma * count_p_i_el(X, sigma, k, el, are_neighbours, N) * (y_i - y_el_k) / ((dist(y_i, y_el_k)) ** 2 * count_6(y_i, y_el_k))

    return suma
