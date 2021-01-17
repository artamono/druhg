"""
Tests for DRUHG clustering algorithm
Shamelessly based on (i.e. ripped off from) the HDBSCAN test code ))
"""
# import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import sparse
from scipy import stats
import pytest

# from sklearn.utils.estimator_checks import check_estimator
# from sklearn.utils.testing import (assert_equal,
#                                    assert_array_equal,
#                                    assert_array_almost_equal,
#                                    assert_raises,
#                                    assert_in,
#                                    assert_not_in,
#                                    assert_no_warnings,
#                                    if_matplotlib)
from druhg import (DRUHG,
                   druhg)

from matplotlib import pyplot as plt

# from sklearn.cluster.tests.common import generate_clustered_data
import sklearn.datasets as datasets
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score

from tempfile import mkdtemp
from functools import wraps

import warnings

moons, _ = datasets.make_moons(n_samples=50, noise=0.05)
blobs, _ = datasets.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])

_plot_graph = 0

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
    labels = dr.relabel(limit1=0, limit2=int(len(XX)/2), fix_outliers=1)
    ari = adjusted_rand_score(iris['target'], labels)
    print ('iris ari', ari)
    assert (ari >= 0.85)

def test_plot_mst():
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50)
    dr.fit(XX)
    dr.minimum_spanning_tree_.plot()


def test_plot_one_dimension():
    iris = datasets.load_iris()
    XX = iris['data']
    XX = XX.reshape(XX.size,1)
    XX = np.array(sorted(XX))
    dr = DRUHG(max_ranking=50)
    dr.fit(XX)
    dr.minimum_spanning_tree_.plot()


def test_2and3():
    cons = 10.
    XX = [[0.,0.],[1.,1.],[cons+3.,2.],[cons+4.,1.],[cons+5.,2.]]
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit1 = 1, limit2 = 1000, verbose=False)
    dr.fit(XX)
    # two clusters
    # assert (len(dr.parents_) == 2)
    print (dr.mst_)
    print (dr.mst_[6]*dr.mst_[7])
    # proper connection between two groups
    assert (dr.mst_[6]*dr.mst_[7] == 2)
    labels = dr.labels_
    print ('pairs', dr.mst_)
    print ('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    assert (labels[0]==labels[1])
    assert (not all(x == labels[0] for x in labels))
    assert (labels[2] == labels[3] == labels[4])
    assert (labels[0] != labels[2])
    assert (n_clusters == 2)
    # assert (1==0)

def test_line():
    XX = [[0.,1.],[0.,2.],[0.,3.],[0.,4.]]
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit1=1, verbose=False)
    dr.fit(XX)
    # starts from the middle
    print (dr.mst_[0], dr.mst_[1])
    print (dr.labels_)
    assert (dr.mst_[0]*dr.mst_[1]==2)
    # zero clusters cause it always grows by 1
    print ('pairs', dr.mst_)
    print ('labels', dr.labels_)
    labels = dr.labels_
    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] == labels[3])
    assert (labels[1] == labels[2])
    # assert (1==0)
# #
def test_longline():
    XX = []
    for i in range(0,100):
        XX.append([0.,i])
    XX = np.array(XX)

    dr = DRUHG(max_ranking=50, limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    # s = 2*len(XX) - 2
    # starts somewhere in the middle
    # and grows one by one
    # that's why there are no clusters
    print ('pairs', dr.mst_)
    print ('labels', dr.labels_)
    # assert (len(dr.parents_)==0)
    labels = dr.labels_
    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] == labels[len(labels)-1])
    assert (labels[1] == labels[len(labels)-2])
    assert (labels[0] != labels[1])
    # assert (0 == 1)

# def test_hypersquare():
#     XX = []
#     size, scale = 6, 1.
#     for i1 in range(0, size):
#         for i2 in range(0, size):
#             for i3 in range(0, size):
#                 for i4 in range(0, size):
#                     for i5 in range(0, size):
#                         XX.append([i1*scale,i2*scale,i3*scale,i4*scale,i5*scale])
#     XX = np.array(XX)
#     dr = DRUHG(max_ranking=10, limit1=1, limit2=len(XX), verbose=False)
#     dr.fit(XX)
#     s = 2*len(XX) - 2
#     print (dr.mst_)
#     print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
#     labels = dr.labels_
#     n_clusters = len(set(labels)) - int(-1 in labels)
#     print('n_clusters', n_clusters)
#     print (dr.mst_)
#     print (XX)
#     print (dr.labels_)
#     assert (n_clusters==1)
#     labels = dr.relabel(limit1=1)
#     n_clusters = len(set(labels)) - int(-1 in labels)
#     print('n_clusters', n_clusters)
#     assert (n_clusters == 5)
#     assert (0==1)

def test_square():
    XX = []
    size, scale = 10, 1
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i,scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=10, limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    print (dr.mst_)
    # print (XX)
    print (dr.labels_)
    # assert (n_clusters==1)
    # labels = dr.relabel(limit1=1, limit2=size*2)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # print('n_clusters', n_clusters)
    print ('pairs', dr.mst_)
    print ('labels', dr.labels_)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_square.png')

    assert (n_clusters >= 5)
    labels = dr.labels_
    assert (not all(x == labels[0] for x in labels))
    assert (labels[1] != labels[size])
    assert (labels[1] != labels[2*size - 1])
    assert (labels[size] != labels[2 * size - 1])
    assert (labels[size] != labels[2 * size - 1])
    assert (labels[1] != labels[len(labels)-2])
    assert (labels[int(size*size/2)] >= 0)


    # assert (0==1)
#

def test_scaled_square():
    XX = []
    size, scale = 10, 0.01
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i,scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit2 = len(XX), verbose=False)
    dr.fit(XX)
    # s = 2*len(XX) - 2
    # print (dr.mst_)
    # print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    print (labels)
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
    dr = DRUHG(max_ranking=200, limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    print (dr.labels_)
    print (dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    # two points are further metrically but close reciprocally
    assert (dr.mst_[s-4]*dr.mst_[s-3] == 0)
    assert (dr.mst_[s-4] + dr.mst_[s-3] == 1)
#
def test_bomb():
    XX = [[0.,1.],[0.,2.],[0.,3.],[0.,4.],[0.,5.]]
    for i in range(-3, 4):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    x = 12
    labs = dr.labels_
    # fuse is separate
    print(labs)
    assert (labs[0]==labs[1]==labs[2]==labs[3])
    assert (np.count_nonzero(labs == labs[0]) == 4)

def test_t():
    XX = []
    for i in range(1, 10):
        XX.append([0.,i])
    for j in range(-9, 10):
        XX.append([j,0.])
    XX = np.array(XX)
    np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print ('n_clusters', n_clusters)
    print ('labels', len(labels), labels)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_t.png')
    # t-center is an outlier too
    assert (n_clusters == 3)
    assert (np.count_nonzero(labels == -1) == 4)

#
def test_cross():
    XX = []
    for j in range(-10, 10):
        XX.append([j,0.])
    for i in range(1, 10):
        XX.append([0., i])
        XX.append([0., i - 10])
    XX = np.array(XX)
    # np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    print (XX)
    # center is an outlier
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print ('n_clusters', n_clusters)
    print ('labels', len(labels), labels)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_cross.png')

    assert (n_clusters == 4)
    assert (np.count_nonzero(labels == -1) == 5)


def test_cube():
    XX = []
    size = 10
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                XX.append([i,j,k])
    XX = np.array(XX)
    # for i, x in enumerate(XX):
    #     print (i, x)
    dr = DRUHG(algorithm='slow', max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print (dr.mst_)
    # assert (0==1)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print ('n_clusters', n_clusters)
    print ('labels', labels)
    assert (n_clusters==1+6)
    # labels = dr.relabel(limit1=1)
    labels = dr.relabel(limit1=1, limit2=len(XX))
    print('out')
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters2', n_clusters)
    print ('labels2', labels)
    assert (n_clusters == 1+6+12)

    # assert (0==1)

def test_druhg_sparse():
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sparse_X = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

    dr = DRUHG()
    dr.fit(sparse_X)
    print ('sparse labels', dr.labels_)

def test_druhg_distance_matrix():
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)

    print (D.shape)
    dr = druhg(D, metric='precomputed')
    print (dr)
    n_clusters = len(set(dr[0])) - int(-1 in dr[0])
    print (n_clusters)


    assert(n_clusters==4)

    dr = DRUHG(metric="precomputed", limit1=5).fit(D)
    labels = dr.labels_
    print (labels)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print (n_clusters)
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_druhg_distance_matrix.png')

    assert(n_clusters==4)

def test_moons_and_blobs():
    XX = X
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    # expecting 4 clusters
    print (labels)

    assert (n_clusters == 4)
#
def test_hdbscan_clusterable_data():
    XX = np.load('druhg\\tests\\clusterable_data.npy')
    dr = DRUHG(max_ranking=50, algorithm='slow', verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    uniques, counts = np.unique(labels, True)
    print (uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print (n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_hdbscan_clusterable_data.png')

    assert (n_clusters==6)
#
def test_chameleon():
    XX = pd.read_csv('druhg\\tests\\chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1 = 1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    values, counts = np.unique(labels, return_counts=True)
    n_clusters = 0
    for i, c in enumerate(counts):
        print (i, c, values[i])
        if c > 500 and values[i] >= 0:
            n_clusters += 1
    print('n_clusters', n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_cham.png')

    dr = DRUHG( max_ranking=200, limit1 = 1, limit2=int(len(XX)/4), exclude=[], verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_cham2.png')

    values, counts = np.unique(dr.labels_, return_counts=True)
    for i, v in enumerate(values):
        if counts[i] > 200:
            print (v, counts[i])

    assert (n_clusters==6)
#
def test_synthetic_outliers():
    XX = pd.read_csv('druhg\\tests\\synthetic.csv', sep=',')
    XX.drop(u'outlier', axis=1, inplace=True)
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit2=len(XX), exclude=[1978,1973], verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_synthetic_outliers.png')

    # values, counts = np.unique(dr.labels_, return_counts=True)
    # for i, v in enumerate(values):
    #     print (v, counts[i])

    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(labels)
    print('n_clusters', n_clusters)
    assert (n_clusters==6)


def test_compound():
    XX = pd.read_csv('druhg/tests/Compound.csv', sep=',', header=None).drop(2, axis=1)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit1=3, limit2=len(XX), exclude=[776], verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig('test_compound.png')

    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # np.save('labels_compound', labels)
    print('n_clusters', n_clusters)
    assert (n_clusters==5)

