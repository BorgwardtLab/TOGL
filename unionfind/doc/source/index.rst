.. unionfind documentation master file, created by
   sphinx-quickstart on Mon Mar  9 22:54:24 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to unionfind's documentation!
=====================================

*unionfind* is a simple module providing a fast Union-Find data structure
implemented in Cython.

The elements of the disjoint set forest are consecutive integer indices -- while
other implementations may allow arbitrary objects to be inserted, restricting
the entries to integers yields the best performance.

.. toctree::
   :maxdepth: 2

.. class:: unionfind.UnionFind(n_points)

   :param n_points: The number of points in the forest.
   :type n_points: int

   .. attribute:: n_sets

      The number of disjoint sets currently in the forest.

   .. method:: find(i)

      Find the root of point *i*.

   .. method:: union(i, j)

      Union the disjoint sets containing *i* and *j*. This method is safe, in
      that if *i* and *j* are already in the same set, nothing will be done.
      Unlike many other implementations, ``union`` will return the index of the
      root of the set resulting from the merger of *i* and *j*.

      :returns: root



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

