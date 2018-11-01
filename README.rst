.. image:: https://img.shields.io/pypi/v/druhg.svg
    :target: https://pypi.python.org/pypi/druhg/
    :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/druhg.svg
    :target: https://github.com/artamono/druhg/blob/master/LICENSE
    :alt: License

=======
DRUHG
=======

| DRUHG – Density Ranking Universal Hierarchical Grouping. Read as droog, it means friend.
| Performs clustering based on even subjective rankings of each datapoint and best stability of a minimum spanning tree of even ranking metric space.
| **Does not require parameters.**
| 
| Tree building and stability determination is taken from `HDBSCAN project. <https://github.com/scikit-learn-contrib/hdbscan>`_ `Thanks Leland McInnes. <https://github.com/lmcinnes/>`_
| It works similarly except HDBSCAN has minpoints parameter, DRUHG does not require any, and gives more freedom in data exploration.
|
| There are no citable publications on this matter, but I would like to create one.
| Even ranking metric tree has a lot of provable attributes and possibly can be used in econometrics.
| So if you are registered as an endorser for the cs.CG (Computational Geometry) subject class of
| arXiv and would like to endorse me, please, follow the link
| https://arxiv.org/auth/endorse?x=VEHO3C

------------------
Basic Concept
------------------

| There are some optional tuning parameters but actual algorithm requires none and is universal.
| It works like **the universal society rule: treat others how you want to be treated**.
| The core of algorithm is to build metric space where distances between two points are even subjective ranking distances of those points. 
|
| Let’s say you have a list of friends and your number one friend is John, but you are number 5 on his friend list, then you would treat him as your number 5 friend.
| After metric space is build it works exactly like HDBSCAN does(minimal spanning tree and it's stability). Based on the papers:
|

    McInnes L, Healy J. *Accelerated Hierarchical Density Based Clustering* 
    In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42.
    2017 `[pdf] <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8215642>`_

    R. Campello, D. Moulavi, and J. Sander, *Density-Based Clustering Based on
    Hierarchical Density Estimates*
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
------------------
How to use DRUHG
------------------
.. code:: python

    import druhg
    from sklearn.datasets import make_blobs
    
    data, _ = make_blobs(1000)
    
    clusterer = druhg.DRUHG()
    cluster_labels = clusterer.fit(data).labels_
It will build the tree and label the points. Now you can condense the tree without rerunning the hardest part of the algorithm. With parameter ``min_samples`` for a cluster size.

.. code:: python
     
    clusterer = clusterer.revisualize(15)
    cluster_labels = clusterer.labels_

-----------
Performance
-----------
| It is a bit slower than the original HDBSCAN.
| But after initial heavy duty run you can do fast cosmetic operations with `.revisualize()` and remove smallest clusters.
|
| There are two optional parameters ``min_ranking`` and ``max_ranking`` that can be used for a better performance.
|
| Let’s go back to John’s example:
| You just found out that your number one friend John has you as number 5 on his list. You confront him, and he tells you that Marry, Ann, Jess and Jill are higher oh his list for obvious reasons. To compromise you and John agrees to treat first five friends evenly as you treat your number 5 friend. 
| That’s ``min_ranking`` parameter(default None). 
|
| Also John proposes that to rank all the neighbors is insane and proposes to rank only first 40 friends, and ignore all others.
| That’s ``max_ranking`` parameter(default None).
| It drastically improves performance!
|
| If ``min_ranking`` is equal to ``max_ranking`` it will be HDBSCAN with ``min_pts`` parameter. 
|
----------
Installing
----------

PyPI install, presuming you have an up to date pip:

.. code:: bash

    pip install druhg


-----------------
Running the Tests
-----------------

The package tests can be run after installation using the command:

.. code:: bash

    nosetests -s druhg

or, if ``nose`` is installed but ``nosetests`` is not in your ``PATH`` variable:

.. code:: bash

    python -m nose -s druhg

The tests may fail :-D

--------------
Python Version
--------------

The druhg library supports both Python 2 and Python 3. 


------------
Contributing
------------

We welcome contributions in any form! Assistance with documentation, particularly expanding tutorials,
is always welcome. To contribute please `fork the project <https://github.com/artamono/druhg/issues#fork-destination-box>`_ 
make your changes and submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.

---------
Licensing
---------

The druhg package is 3-clause BSD licensed.
