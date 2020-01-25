"""
Tests for DRUHG clustering algorithm
Shamelessly based on (i.e. ripped off from) the HDBSCAN test code ))
"""
# import pickle
from nose.tools import assert_less
from nose.tools import assert_greater_equal
from nose.tools import assert_not_equal
import numpy as np
import pandas as pd
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
                   druhg)

# from sklearn.cluster.tests.common import generate_clustered_data
import sklearn.datasets as datasets
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score

from tempfile import mkdtemp
from functools import wraps
from nose import SkipTest

import warnings

moons, _ = datasets.make_moons(n_samples=50, noise=0.05)
blobs, _ = datasets.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])

def test_iris():
    iris = datasets.load_iris()
    XX = iris['data']
    # print (XX, type(XX))
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    ari = adjusted_rand_score(iris['target'], labels)
    print ('iris ari', ari)
    assert (ari >= 0.50)
    # breaking biggest cluster
    labels = dr.relabel(limit1=0, limit2=len(XX)/2, fix_outliers=1)
    ari = adjusted_rand_score(iris['target'], labels)
    print ('iris ari', ari)
    assert (ari >= 0.85)

def test_plot_mst():
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50)
    dr.fit(XX)
    dr.minimum_spanning_tree_.plot()

def test_2and3():
    XX = [[0.,0.],[1.,1.],[3.,2.],[4.,1.],[5.,2.]]
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit1 = 1, limit2 = 1000, verbose=False)
    dr.fit(XX)
    # two clusters
    assert (len(dr.parents_) == 2)
    print (dr.mst_)
    print (dr.mst_[6]*dr.mst_[7])
    # proper connection between two groups
    assert (dr.mst_[6]*dr.mst_[7] == 2)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    assert (n_clusters == 2)

def test_line():
    XX = [[0.,1.],[0.,2.],[0.,3.],[0.,4.]]
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    # starts from the middle
    print (dr.mst_[0], dr.mst_[1])
    assert (dr.mst_[0]*dr.mst_[1]==2)
    # zero clusters cause it always grows by 1
    print ('clusters', len(dr.parents_))
    assert (len(dr.parents_)==0)
    # assert (1==0)
#
def test_longline():
    XX = []
    for i in range(0,1000):
        XX.append([0.,i])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    # s = 2*len(XX) - 2
    # starts somewhere in the middle
    # and grows one by one
    # that's why there are no clusters
    print ('pairs', dr.mst_)
    print ('parents', dr.parents_)
    assert (len(dr.parents_)==0)
    # assert (0 == 1)

def test_square():
    XX = []
    size, scale = 6, 1
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i,scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    assert (n_clusters==1)
    labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    assert (n_clusters == 5)

def test_scaled_square():
    XX = []
    size, scale = 10, 3
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i,scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit2 = size**3, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print ('n_clusters', n_clusters)
    assert (n_clusters==1)

def test_two_squares():
    XX = []
    size, scale = 6, 1
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i, scale*j])
            XX.append([2*size + scale*i, scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert (n_clusters==2)

def test_particles():
    XX = [[-0.51,1.5], [1.51,1.5]]
    for i in range(-3, 5):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    # two points are further metrically but close reciprocally
    assert (dr.mst_[s-4]*dr.mst_[s-3] == 0)
    assert (dr.mst_[s-4] + dr.mst_[s-3] == 1)
    # assert (0==1)
#
def test_bomb():
    XX = [[0.,1.],[0.,2.],[0.,3.],[0.,4.],[0.,5.]]
    for i in range(-3, 4):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.parents_)
    x = 12
    print (dr.mst_[s - 1-x], dr.mst_[s - 2-x], XX[dr.mst_[s - 1-x]], XX[dr.mst_[s - 2-x]])
    assert (dr.mst_[s - 1-x]+dr.mst_[s - 2-x] == 32)
    assert (dr.mst_[s - 1-x]*dr.mst_[s - 2-x] == 0)
    assert (len(dr.parents_)==7)
    # assert (0==1)

def test_t():
    XX = []
    for i in range(1, 10):
        XX.append([0.,i])
    for j in range(-10, 10):
        XX.append([j,0.])
    XX = np.array(XX)
    np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    print (XX)
    # center will be randomly assigned to one of the legs
    assert (len(dr.parents_) == 3)

def test_cross():
    XX = []
    for i in range(1, 10):
        XX.append([0., i])
        XX.append([0., i - 10])
    for j in range(-10, 10):
        XX.append([j,0.])
    XX = np.array(XX)
    np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    print (XX)
    # center is an outlier
    assert (len(dr.parents_)==4)

def test_cube():
    XX = []
    size = 8
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                XX.append([i,j,k])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (len(dr.parents_), dr.parents_)
    # assert (0==1)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print ('n_clusters', n_clusters)
    print ('labels', labels)
    assert (n_clusters==1+6)
    labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    assert (n_clusters == 1+6+12)

def test_chameleon():
    XX = pd.read_csv('druhg\\tests\\chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    assert (n_clusters==6)

def test_druhg_sparse():
    sparse_X = sparse.csr_matrix(X)
    print ('shapes', X.shape, sparse_X.shape)
    print (type(sparse_X))
    print ('sparse_X')
    print (sparse_X)
    print (X)
    DRUHG().fit(sparse_X)
    # assert (0 == 1)

def test_druhg_distance_matrix():
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)

    print (D.shape)
    dt = druhg(D, metric='precomputed')
    # number of clusters, ignoring noise if present
    # n_clusters_1 = len(set(labels)) - int(-1 in labels)  # ignore noise
    # assert_equal(n_clusters_1, n_clusters)

    labels = DRUHG(metric="precomputed").fit(D).labels_
    # n_clusters_2 = len(set(labels)) - int(-1 in labels)
    # assert_equal(n_clusters_2, n_clusters)
    # assert (0==1)

    # validity = validity_index(D, labels, metric='precomputed', d=2)
    # assert_greater_equal(validity, 0.6)

def test_moons_and_blobs():
    XX = X
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    # expecting 4 clusters
    assert (n_clusters == 4)

def test_hdbscan_clusterable_data():
    XX = np.load('druhg\\tests\\clusterable_data.npy')
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    uniques, counts = np.unique(labels, True)
    print (uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print (n_clusters)
    # expecting 6 big clusters
    assert (n_clusters==6)
