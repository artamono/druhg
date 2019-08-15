.. image:: https://img.shields.io/pypi/v/druhg.svg
    :target: https://pypi.python.org/pypi/druhg/
    :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/druhg.svg
    :target: https://github.com/artamono/druhg/blob/master/LICENSE
    :alt: License

=====
WORK IN PROGRESS: DRUHG
=====

| DRUHG – Density Ranking Universal Hierarchical Grouping. Друг - read as droog, it means friend.
| Performs clustering based on even subjective rankings of each datapoint and best stability of a minimum spanning tree of even ranking metric space.
| **Does not require parameters.**
| 
| Builds hierarchical tree based on similarities of "density".
| 
| There are no citable publications on this matter, but I would like to create one.
| If you are registered as an endorser for the cs.CG (Computational Geometry) subject class of
| arXiv and would like to endorse me, please, follow the link
| https://arxiv.org/auth/endorse?x=VEHO3C

-------------
Basic Concept
-------------

| The algorithm is based on **the universal society rule: treat others how you want to be treated**.
|
| Let’s say you have a list of friends and your number one friend is John, but you are number 5 on his friend list, then you would treat him as your number 5 friend.
| This relationship will create a lot of data. 
| After that you can find optimal relationship fo each object and add them up to construct the tree.
| It uses knn queries where k is productivity parameter.

----------------
WIP: How to use DRUHG
----------------
.. code:: python

    import druhg
    from sklearn.datasets import make_blobs
    
    data, _ = make_blobs(1000)
    
    clusterer = druhg.DRUHG()
    cluster_labels = clusterer.fit(data).labels_

-----------
Performance
-----------
| It is a bit slow compared to other algorithms but you can run it only once.
| No parameters - no reruns.
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

Contributions in any form are welcomed! Assistance with documentation is always welcome. To contribute please `fork the project <https://github.com/artamono/druhg/issues#fork-destination-box>`_ 
make your changes and submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.

---------
Licensing
---------

The druhg package is 3-clause BSD licensed.
