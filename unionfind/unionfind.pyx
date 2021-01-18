cimport cython
from libc.stdlib cimport malloc, free
import itertools

class MultipleUnionFind:
    def __init__(self,n_uf,n):
        """
        n_uf is the number of unionfind objects
        n is the number of elements in each unionfind
        """
        self.uflist = [UnionFind(n) for i in range(n_uf)]

    def find(self,values):
        """
        values in a vector of length n
        """
        return [uf.find(values[i]) for i,uf in enumerate(self.uflist)]

    def union(self,values1, values2, mask):
        return [uf.union(values1[i],values2[i]) for i,uf in enumerate(self.uflist) if mask[i]]

    def roots(self):
        ### returns -1 when the list of one uf is exhausted
        return itertools.zip_longest(*[uf.roots() for uf in self.uflist],fillvalue = -1)


cdef class UnionFind:
    cdef int n_points
    cdef int * parent
    cdef int * rank
    cdef int _n_sets

    def __cinit__(self, n_points):
        self.n_points = n_points
        self.parent = <int *> malloc(n_points * sizeof(int))
        self.rank = <int *> malloc(n_points * sizeof(int))

        cdef int i
        for i in range(n_points):
            self.parent[i] = i

        self._n_sets = n_points

    def __dealloc__(self):
        free(self.parent)
        free(self.rank)

    cdef int _find(self, int i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]
    
    def roots(self):
        for vertex in range(self.n_points):
            if vertex == self.parent[vertex]:
                yield vertex

    def find(self, int i):
        if (i < 0) or (i > self.n_points):
            raise ValueError("Out of bounds index.")
        return self._find(i)

    def merge(self, int u, int v):
        cdef int root_u, root_v
        if u!= v:
            root_u = self.find(u)
            root_v = self.find(v)
            self.parent[root_u] = root_v

    def union(self, int i, int j):
        if (i < 0) or (i > self.n_points) or (j < 0) or (j > self.n_points):
            raise ValueError("Out of bounds index.")

        cdef int root_i, root_j
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self._n_sets -= 1
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                return root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                return root_i
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
                return root_j
        else:
            return root_i

    property n_sets:
        def __get__(self):
            return self._n_sets
