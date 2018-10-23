"""
Tests for DRUHG clustering algorithm
Shamelessly based on (i.e. ripped off from) the HDBSCAN test code ))
"""
# import pickle
from nose.tools import assert_less
from nose.tools import assert_greater_equal
from nose.tools import assert_not_equal
import numpy as np
from scipy.spatial import distance
from scipy import sparse
from scipy import stats
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (assert_equal,
                                   assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_raises,
                                   assert_in,
                                   assert_not_in,
                                   assert_no_warnings,
                                   if_matplotlib)
from druhg import (DRUHG,
                     druhg,
                     validity_index)
# from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

from tempfile import mkdtemp
from functools import wraps
from nose import SkipTest

from sklearn import datasets

import warnings

n_clusters = 3
# X = generate_clustered_data(n_clusters=n_clusters, n_samples_per_cluster=50)
X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)


def if_pandas(func):
    """Test decorator that skips test if pandas not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import pandas
        except ImportError:
            raise SkipTest('Pandas not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def if_networkx(func):
    """Test decorator that skips test if networkx not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import networkx
        except ImportError:
            raise SkipTest('NetworkX not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def generate_noisy_data():
    blobs, _ = datasets.make_blobs(n_samples=200,
                                   centers=[(-0.75, 2.25), (1.0, 2.0)],
                                   cluster_std=0.25)
    moons, _ = datasets.make_moons(n_samples=200, noise=0.05)
    noise = np.random.uniform(-1.0, 3.0, (50, 2))
    return np.vstack([blobs, moons, noise])


def homogeneity(labels1, labels2):
    num_missed = 0.0
    for label in set(labels1):
        matches = labels2[labels1 == label]
        match_mode = mode(matches)[0][0]
        num_missed += np.sum(matches != match_mode)

    for label in set(labels2):
        matches = labels1[labels2 == label]
        match_mode = mode(matches)[0][0]
        num_missed += np.sum(matches != match_mode)

    return num_missed / 2.0


def test_druhg_distance_matrix():
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)

    labels, p, persist, ctree, ltree, mtree = druhg(D, metric='precomputed')
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
    assert_equal(n_clusters_1, n_clusters)

    labels = DRUHG(metric="precomputed").fit(D).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

    validity = validity_index(D, labels, metric='precomputed', d=2)
    assert_greater_equal(validity, 0.6)


def test_druhg_distance_matrix_revisualize():
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)

    labels, p, persist, ctree, ltree, mtree = druhg(D, metric='precomputed')
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
    assert_equal(n_clusters_1, n_clusters)

    droog = DRUHG(metric="precomputed").fit(D)

    labels = droog.labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

    validity = validity_index(D, labels, metric='precomputed', d=2)
    assert_greater_equal(validity, 0.6)

    droog.revisualize(min_samples = 120)
    labels = droog.labels_
    n_clusters_3 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_3, n_clusters)

    droog.revisualize(min_samples = 2)
    labels = droog.labels_
    n_clusters_4 = len(set(labels)) - int(-1 in labels)
    assert_not_equal(n_clusters_4,n_clusters_3)

    validity = validity_index(D, labels, metric='precomputed', d=2)
    assert_greater_equal(validity, 0.6)

#def test_druhg_sparse_distance_matrix():
#    D = distance.squareform(distance.pdist(X))
#    D /= np.max(D)
#
#    threshold = stats.scoreatpercentile(D.flatten(), 50)
#
#    D[D >= threshold] = 0.0
#    D = sparse.csr_matrix(D)
#    D.eliminate_zeros()
#
#    labels, p, persist, ctree, ltree, mtree = druhg(D, metric='precomputed')
#    # number of clusters, ignoring noise if present
#    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
#    assert_equal(n_clusters_1, n_clusters)
#
#    labels = DRUHG(metric="precomputed",
#                     gen_min_span_tree=True).fit(D).labels_
#    n_clusters_2 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_2, n_clusters)
#
#
#def test_druhg_feature_vector():
#    labels, p, persist, ctree, ltree, mtree = druhg(X)
#    n_clusters_1 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_1, n_clusters)
#
#    labels = DRUHG().fit(X).labels_
#    n_clusters_2 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_2, n_clusters)
#
#    validity = validity_index(X, labels)
#    assert_greater_equal(validity, 0.4)
#
#
def test_druhg_boruvka_kdtree():
    labels, p, persist, ctree, ltree, mtree = druhg(
        X, algorithm='boruvka_kdtree')
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    labels = DRUHG(algorithm='boruvka_kdtree',
                     gen_min_span_tree=True).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

    assert_raises(ValueError,
                  druhg,
                  X,
                  algorithm='boruvka_kdtree',
                  metric='russelrao')


def test_druhg_boruvka_tree():
    labels, p, persist, ctree, ltree, mtree = druhg(
        X, algorithm='boruvka_balltree')
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    labels = DRUHG(algorithm='boruvka_balltree',
                     gen_min_span_tree=True).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

    assert_raises(ValueError,
                  druhg,
                  X,
                  algorithm='boruvka_balltree',
                  metric='cosine')


def test_druhg_generic():
    labels, p, persist, ctree, ltree, mtree = druhg(X, algorithm='generic')
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    labels = DRUHG(algorithm='generic',
                     gen_min_span_tree=True).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)

#
#def test_druhg_high_dimensional():
#    H, y = make_blobs(n_samples=50, random_state=0, n_features=64)
#    # H, y = shuffle(X, y, random_state=7)
#    H = StandardScaler().fit_transform(H)
#    labels, p, persist, ctree, ltree, mtree = druhg(H)
#    n_clusters_1 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_1, n_clusters)
#
#    labels = DRUHG(algorithm='best', metric='seuclidean',
#                     V=np.ones(H.shape[1])).fit(H).labels_
#    n_clusters_2 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_2, n_clusters)
#
#
def test_druhg_best_tree_metric():
    labels, p, persist, ctree, ltree, mtree = druhg(X, metric='seuclidean',
                                                      V=np.ones(X.shape[1]))
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    labels = DRUHG(metric='seuclidean', V=np.ones(X.shape[1])).fit(X).labels_
    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)
#
#
#def test_druhg_callable_metric():
#    # metric is the function reference, not the string key.
#    metric = distance.euclidean
#
#    labels, p, persist, ctree, ltree, mtree = druhg(X, metric=metric)
#    n_clusters_1 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_1, n_clusters)
#
#    labels = DRUHG(metric=metric).fit(X).labels_
#    n_clusters_2 = len(set(labels)) - int(-1 in labels)
#    assert_equal(n_clusters_2, n_clusters)
#
#
#def test_druhg_input_lists():
#    X = [[1., 2.], [3., 4.]]
#    DRUHG().fit(X)  # must not raise exception
#

def test_druhg_boruvka_kdtree_matches():

    data = generate_noisy_data()

    labels_true, p, persist, ctree, ltree, mtree = druhg(
        data, algorithm='generic')
    labels_boruvka, p, persist, ctree, ltree, mtree = druhg(
        data, algorithm='boruvka_kdtree')

    num_mismatches = homogeneity(labels_true, labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.15)

    labels_true = DRUHG(algorithm='generic').fit_predict(data)
    labels_boruvka = DRUHG(algorithm='boruvka_kdtree').fit_predict(data)

    num_mismatches = homogeneity(labels_true, labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.15)


def test_druhg_boruvka_balltree_matches():

    data = generate_noisy_data()

    min_samples = 3
    max_ranking = 32

    labels_true, p, persist, ctree, ltree, mtree = druhg(
        data, algorithm='generic', min_samples = min_samples)
    labels_boruvka, p, persist, ctree, ltree, mtree = druhg(
        data, algorithm='boruvka_balltree', min_samples = min_samples, max_ranking = max_ranking)

    num_mismatches = homogeneity(labels_true, labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.15)

    labels_true = DRUHG(algorithm='generic', min_samples = min_samples).fit_predict(data)
    labels_boruvka = DRUHG(algorithm='boruvka_kdtree', min_samples = min_samples, max_ranking = max_ranking).fit_predict(data)

    num_mismatches = homogeneity(labels_true, labels_boruvka)

    assert_less(num_mismatches / float(data.shape[0]), 0.15)


#def test_condensed_tree_plot():
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#    if_matplotlib(clusterer.condensed_tree_.plot)(
#        select_clusters=True,
#        label_clusters=True,
#        selection_palette=('r', 'g', 'b'),
#        cmap='Reds')
#    if_matplotlib(clusterer.condensed_tree_.plot)(log_size=True,
#                                                  colorbar=False,
#                                                  cmap='none')


#def test_single_linkage_tree_plot(#):
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#    if_matplotlib(clusterer.single_linkage_tree_.plot)(cmap='Reds')
#    if_matplotlib(clusterer.single_linkage_tree_.plot)(vary_line_width=False,
#                                                       truncate_mode='lastp',
#                                                       p=10, cmap='none',
#                                                       colorbar=False)
#
#
#def test_min_span_tree_plot():
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#    if_matplotlib(clusterer.minimum_spanning_tree_.plot)(edge_cmap='Reds')
#
#    H, y = make_blobs(n_samples=50, random_state=0, n_features=10)
#    H = StandardScaler().fit_transform(H)
#
#    clusterer = DRUHG(gen_min_span_tree=True).fit(H)
#    if_matplotlib(clusterer.minimum_spanning_tree_.plot)(edge_cmap='Reds',
#                                                         vary_line_width=False,
#                                                         colorbar=False)
#
#    H, y = make_blobs(n_samples=50, random_state=0, n_features=40)
#    H = StandardScaler().fit_transform(H)
#
#    clusterer = DRUHG(gen_min_span_tree=True).fit(H)
#    if_matplotlib(clusterer.minimum_spanning_tree_.plot)(edge_cmap='Reds',
#                                                         vary_line_width=False,
#                                                         colorbar=False)
#
#
#def test_tree_numpy_output_formats():
#
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#
#    clusterer.single_linkage_tree_.to_numpy()
#    clusterer.condensed_tree_.to_numpy()
#    clusterer.minimum_spanning_tree_.to_numpy()
#
#
#def test_tree_pandas_output_formats():
#
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#    if_pandas(clusterer.condensed_tree_.to_pandas)()
#    if_pandas(clusterer.single_linkage_tree_.to_pandas)()
#    if_pandas(clusterer.minimum_spanning_tree_.to_pandas)()
#
#
#def test_tree_networkx_output_formats():
#
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#    if_networkx(clusterer.condensed_tree_.to_networkx)()
#    if_networkx(clusterer.single_linkage_tree_.to_networkx)()
#    if_networkx(clusterer.minimum_spanning_tree_.to_networkx)()
#
#
#def test_druhg_outliers():
#    clusterer = DRUHG(gen_min_span_tree=True).fit(X)
#    scores = clusterer.outlier_scores_
#    assert scores is not None
#

# def test_druhg_unavailable_attributes():
#     clusterer = DRUHG(gen_min_span_tree=False)
#     with warnings.catch_warnings(record=True) as w:
#         tree = clusterer.condensed_tree_
#         assert len(w) > 0
#         assert tree is None
#     with warnings.catch_warnings(record=True) as w:
#         tree = clusterer.single_linkage_tree_
#         assert len(w) > 0
#         assert tree is None
#     with warnings.catch_warnings(record=True) as w:
#         scores = clusterer.outlier_scores_
#         assert len(w) > 0
#         assert scores is None
#     with warnings.catch_warnings(record=True) as w:
#         tree = clusterer.minimum_spanning_tree_
#         assert len(w) > 0
#         assert tree is None


# def test_druhg_min_span_tree_availability():
#     clusterer = DRUHG().fit(X)
#     tree = clusterer.minimum_spanning_tree_
#     assert tree is None
#     D = distance.squareform(distance.pdist(X))
#     D /= np.max(D)
#     DRUHG(metric='precomputed').fit(D)
#     tree = clusterer.minimum_spanning_tree_
#     assert tree is None

#def test_druhg_approximate_predict():
#    clusterer = DRUHG(prediction_data=True).fit(X)
#    cluster, prob = approximate_predict(clusterer, np.array([[-1.5, -1.0]]))
#    assert_equal(cluster, 2)
#    cluster, prob = approximate_predict(clusterer, np.array([[1.5, -1.0]]))
#    assert_equal(cluster, 1)
#    cluster, prob = approximate_predict(clusterer, np.array([[0.0, 0.0]]))
#    assert_equal(cluster, -1)

# def test_druhg_membership_vector():
#     clusterer = DRUHG(prediction_data=True).fit(X)
#     vector = membership_vector(clusterer, np.array([[-1.5, -1.0]]))
#     assert_array_almost_equal(
#         vector,
#         np.array([[ 0.05705305,  0.05974177,  0.12228153]]))
#     vector = membership_vector(clusterer, np.array([[1.5, -1.0]]))
#     assert_array_almost_equal(
#         vector,
#         np.array([[ 0.09462176,  0.32061556,  0.10112905]]))
#     vector = membership_vector(clusterer, np.array([[0.0, 0.0]]))
#     assert_array_almost_equal(
#         vector,
#         np.array([[ 0.03545607,  0.03363318,  0.04643177]]))
#
# def test_druhg_all_points_membership_vectors():
#     clusterer = DRUHG(prediction_data=True).fit(X)
#     vects = all_points_membership_vectors(clusterer)
#     assert_array_almost_equal(vects[0], np.array([7.86400992e-002,
#                                                    2.52734246e-001,
#                                                    8.38299608e-002]))
#     assert_array_almost_equal(vects[-1], np.array([8.09055344e-001,
#                                                    8.35882503e-002,
#                                                    1.07356406e-001]))


##def test_druhg_all_points_membership_vectors():
##    clusterer = DRUHG(prediction_data=True, max_ranking=200).fit(X)
##    vects = all_points_membership_vectors(clusterer)
##    assert_array_equal(vects,
##                       np.zeros(clusterer.prediction_data_.raw_data.shape[0]))

##
def test_druhg_badargs():
    assert_raises(ValueError,
                  druhg,
                  X='fail')
    assert_raises(ValueError,
                  druhg,
                  X=None)
    assert_raises(ValueError,
                  druhg,
                  X, max_ranking='fail')
    assert_raises(ValueError,
                  druhg,
                  X, min_samples='fail')
    assert_raises(ValueError,
                  druhg,
                  X, min_samples=-1)
    assert_raises(ValueError,
                  druhg,
                  X, metric='imperial')
    assert_raises(ValueError,
                  druhg,
                  X, metric=None)
    assert_raises(ValueError,
                  druhg,
                  X, metric='minkowski', p=-1)
#    assert_raises(ValueError,
#                  druhg,
#                  X, metric='minkowski', p=-1, algorithm='prims_kdtree')
#    assert_raises(ValueError,
#                  druhg,
#                  X, metric='minkowski', p=-1, algorithm='prims_balltree')
    assert_raises(ValueError,
                  druhg,
                  X, metric='minkowski', p=-1, algorithm='boruvka_balltree')
    assert_raises(ValueError,
                  druhg,
                  X, metric='precomputed', algorithm='boruvka_kdtree')
#    assert_raises(ValueError,
#                  druhg,
#                  X, metric='precomputed', algorithm='prims_kdtree')
#    assert_raises(ValueError,
#                  druhg,
#                  X, metric='precomputed', algorithm='prims_balltree')
    assert_raises(ValueError,
                  druhg,
                  X, metric='precomputed', algorithm='boruvka_balltree')
    assert_raises(ValueError,
                  druhg,
                  X, alpha=-1)
    assert_raises(ValueError,
                  druhg,
                  X, alpha='fail')
    assert_raises(Exception,
                  druhg,
                  X, algorithm='something_else')
    assert_raises(TypeError,
                  druhg,
                  X, metric='minkowski', p=None)
    assert_raises(ValueError,
                  druhg,
                  X, leaf_size=0)


def test_druhg_sparse():

    sparse_X = sparse.csr_matrix(X)

    labels = DRUHG().fit(sparse_X).labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters, 3)


def test_druhg_caching():

    cachedir = mkdtemp()
    labels1 = DRUHG(memory=cachedir, min_samples=5).fit(X).labels_
    labels2 = DRUHG(memory=cachedir, min_samples=5,
                      max_ranking=6).fit(X).labels_
    n_clusters1 = len(set(labels1)) - int(-1 in labels1)
    n_clusters2 = len(set(labels2)) - int(-1 in labels2)
    assert_equal(n_clusters1, n_clusters2)


def test_druhg_is_sklearn_estimator():

    check_estimator(DRUHG)

def test_druhg_multiple_runs_distance_matrix():
    D = distance.squareform(distance.pdist(X[:25]))
    D /= np.max(D)
    n_clusters2 = 3

    min_samples = 5

    run_times = 0
    labels, p, persist, ctree, ltree, mtree = druhg(D, metric='precomputed', run_times=run_times, min_samples = min_samples)
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
    assert_equal(n_clusters_1, n_clusters2)
    assert_equal(int(-1 in labels), 0)

    run_times = 1
    labels, p, persist, ctree, ltree, mtree = druhg(D, metric='precomputed', run_times=run_times, min_samples = min_samples)
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
    assert_equal(n_clusters_1, n_clusters2)
    assert_equal(int(-1 in labels), 0)

    run_times = 2
    labels, p, persist, ctree, ltree, mtree = druhg(D, metric='precomputed', run_times=run_times, min_samples = min_samples)
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
    assert_equal(n_clusters_1, n_clusters2)
    assert_equal(int(-1 in labels), 1)

    validity = validity_index(D, labels, metric='precomputed', d=2)
    assert_greater_equal(validity, 0.6)


# Probably not applicable now #
# def test_dbscan_sparse():
# def test_dbscan_balltree():
# def test_pickle():
# def test_dbscan_core_samples_toy():
# def test_boundaries():
