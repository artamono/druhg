# -*- coding: utf-8 -*-
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD


import numpy as np

# from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from warnings import warn

class SingleLinkage(object):
# todo: make it pretty
    def __init__(self, mst_pairs, values, labels):
        self._mst_pairs = mst_pairs
        self._values = values
        # self._data = data
        self._labels = labels

    def fast_find(self, unionfind, n):
        n = int(n)
        p = unionfind[n]
        if p == 0:
            return n

        while unionfind[p] != 0:
            p = unionfind[p]

        # label up to the root
        while p != n:
            temp = unionfind[n]
            unionfind[n] = p
            n = temp

        return p


    def draw_dendrogram(self, ax, pairs, values, labels, lw=20., alpha=0.4, cmap='viridis'):
        try:
            from matplotlib import collections as mc
            from matplotlib.pyplot import Arrow
            from matplotlib.pyplot import Normalize
            from matplotlib.pyplot import cm
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        min_index, max_index = min(pairs), max(pairs)
        if min_index < 0:
            raise ValueError('Indices should be non-negative')

        size = int(len(pairs) / 2 + 1)

        union_size = size
        if max_index > union_size - 1:
            union_size = max_index + 1
        union_size += 2

        # we will create Union Find as usual
        uf, sz = np.zeros(2 * union_size, dtype=int), np.ones(union_size)
        next_label = union_size + 1
        # also we need links
        l, r = np.arange(0, 2*union_size), np.arange(0, 2*union_size)

        next_label = union_size + 1
        for j in range(0, size - 1):
            a, b = pairs[2 * j], pairs[2 * j + 1]

            # we will stack first cluster on the left of second
            aa = a
            while aa != r[aa]:
                aa = r[aa]
            bb = b
            while bb != l[bb]:
                bb = l[bb]
            l[bb] = aa
            r[aa] = bb # linking


            aa = a
            while aa != l[aa]:
                aa = l[aa]
            bb = b
            while bb != r[bb]:
                bb = r[bb]
            l[next_label] = aa # marking the borders
            r[next_label] = bb

            aa, bb = self.fast_find(uf, a), self.fast_find(uf, b)
            uf[aa] = uf[bb] = next_label

            # i = next_label - union_size
            # a2 = (uf[a] != 0) * (aa - union_size)
            # b2 = (uf[b] != 0) * (bb - union_size)
            # na, nb = sz[a2], sz[b2]
            # sz[i] = na + nb

            next_label += 1

        x_arr = self.arrange_nodes_on_x_axis(uf, union_size, l, r, 200.)

        norm = len(np.unique(pairs))
        sm = cm.ScalarMappable(cmap=cmap,
                                   norm=Normalize(0, norm))
        sm.set_array(norm)

        colors = self.get_dendro_colors(labels)
        heights = {}
        uf.fill(0)
        next_label = union_size + 1
        for j in range(0, size - 1):
            v = np.log2(1. + values[j]) # logarithm
            heights[next_label] = v

            a, b = pairs[2 * j], pairs[2 * j + 1]

            # i = next_label - union_size
            aa, bb = self.fast_find(uf, a), self.fast_find(uf, b)
            x_arr[next_label] = (x_arr[r[aa]] + x_arr[l[bb]])/2.
            uf[aa] = uf[bb] = next_label
            next_label += 1

            # a = (uf[a] != 0) * (aa - union_size)
            # b = (uf[b] != 0) * (bb - union_size)
            # na, nb = sz[a], sz[b]
            # sz[i] = na + nb

            ha, hb = 0, 0
            xa, xb = x_arr[aa], x_arr[bb]
            if aa in heights:
                ha = heights[aa]
            if bb in heights:
                hb = heights[bb]

            c = 'gray'
            if labels[a] == labels[b] and labels[a] > 0:
                c = colors[labels[a]]

            ax.plot([xa, xa], [ha, v], color=c)
            ax.plot([xb, xb], [hb, v], color=c)
            ax.plot([xa, xb], [v, v], color=c)

        ax.set_xticks([])
        for side in ('right', 'top', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.set_ylabel('distance')

        # line_collection.set_array(self._mst[:, 2].T)
        return ax

    def arrange_nodes_on_x_axis(self, uf, union_size, l, r, step = 2.):
        x_arr = np.zeros(2 * union_size, dtype=int)
        processed = {}
        x = 1.
        # constructing the tree
        # there is a possibility of a forest instead of a tree
        for a in (0, union_size):
            if l[a] == a: # can happened when some indices were not passed
                continue
            aa = self.fast_find(uf, a)
            if aa in processed:
                continue
            processed[aa] = aa
            aa = a
            while aa != r[aa]:
                aa = r[aa]
            while aa != l[aa]:
                x_arr[aa] = x
                x += step
                aa = l[aa]
            x_arr[aa] = x

        return x_arr

    def get_dendro_colors(self, labels):
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError('You must install the seaborn library to draw colored labels.')

        unique, counts = np.unique(labels, return_counts=True)
        sorteds = np.argsort(counts)
        s = len(sorteds)

        i = sorteds[s - 1]
        max_size = counts[i]
        if unique[i] < 0:
            max_size = counts[sorteds[s - 2]]

        color_map = {}
        palette = sns.color_palette('bright', s + 1)
        col = 0
        a = (1. - 0.3) / (max_size - 1)
        b = 0.3 - a
        while s:
            s -= 1
            i = sorteds[s]
            if unique[i] < 0:  # outliers
                color_map[unique[i]] = (0., 0., 0., 0.15)
                continue
            alpha = a * counts[i] + b
            color_map[unique[i]] = palette[col] + (alpha,)
            col += 1

        return color_map

    def plot(self, axis=None):
        """Plot the dendrogram.

        Parameters
        ----------

        axis : matplotlib axis, optional
               The axis to render the plot to

        Returns
        -------

        axis : matplotlib axis
                The axis used the render the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        if axis is None:
            axis = plt.gca()
            axis.set_axis_off()


        axis = self.draw_dendrogram(axis, self._mst_pairs, self._values, self._labels)

        return axis


class MinimumSpanningTree(object):
    def __init__(self, mst_pairs, data, labels):
        self._mst_pairs = mst_pairs
        self._data = data
        self._labels = labels

    def decrease_dimensions(self):
        if self._data.shape[1] > 2:
            # Get a 2D projection; if we have a lot of dimensions use PCA first
            if self._data.shape[1] > 32:
                # Use PCA to get down to 32 dimension
                data_for_projection = PCA(n_components=32).fit_transform(self._data)
            else:
                data_for_projection = self._data

            projection = TSNE().fit_transform(data_for_projection)
        elif self._data.shape[1] == 2:
            projection = self._data.copy()
        else:
            # one dimensional. We need to add dimension
            projection = self._data.copy()
            projection = np.array([e for e in enumerate(projection)], np.int)

        return projection

    def get_node_colors(self, labels):
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError('You must install the seaborn library to draw colored labels.')

        unique, counts = np.unique(labels, return_counts=True)
        sorteds = np.argsort(counts)
        s = len(sorteds)

        i = sorteds[s - 1]
        max_size = counts[i]
        if unique[i] < 0:
            max_size = counts[sorteds[s - 2]]

        color_map = {}
        palette = sns.color_palette('bright', s + 1)
        col = 0
        a = (1. - 0.3) / (max_size - 1)
        b = 0.3 - a
        while s:
            s -= 1
            i = sorteds[s]
            if unique[i] < 0:  # outliers
                color_map[unique[i]] = (0., 0., 0., 0.15)
                continue
            alpha = a * counts[i] + b
            color_map[unique[i]] = palette[col] + (alpha,)
            col += 1

        return [color_map[x] for x in labels]

    def draw_simple_edges(self, ax, pairs, pos, lw=2., alpha=0.5):
        try:
            from matplotlib import collections as mc
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        lines, line_heads = [], []

        size = int(len(pairs) / 2)
        for i in range(0, size):
            start, end = pos[pairs[2 * i]], pos[pairs[2 * i + 1]]

            lines.append([start, end])
            line_heads.append([((start + end) / 2 + end) / 2, end])

        lc = mc.LineCollection(lines, colors=(0., 0., 0., alpha), linewidths=lw)
        ax.add_collection(lc)

        lc = mc.LineCollection(line_heads, colors=(0., 0., 0., alpha * 0.8), linewidths=lw * 4.)
        ax.add_collection(lc)

    def fast_find(self, unionfind, n):
        n = int(n)
        p = unionfind[n]
        if p == 0:
            return n

        while unionfind[p] != 0:
            p = unionfind[p]

        # label up to the root
        while p != n:
            temp = unionfind[n]
            unionfind[n] = p
            n = temp

        return p

    def merge_means(self, na, meana, nb, meanb):
        delta = meanb - meana
        meana = meana + delta * nb / (na + nb)
        return meana

    def draw_druhg_edges(self, ax, pairs, pos, lw=20., alpha=0.4):
        try:
            from matplotlib import collections as mc
            from matplotlib.pyplot import Arrow
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        min_index, max_index = min(pairs), max(pairs)
        if min_index < 0:
            raise ValueError('Indices should be non-negative')

        size = int(len(pairs) / 2 + 1)

        union_size = size
        if max_index > union_size - 1:
            union_size = max_index + 1
        union_size += 2

        uf, sz = np.zeros(2 * union_size, dtype=int), np.ones(union_size)
        next_label = union_size + 1

        default_color = (0, 0, 0, alpha)
        min_arrow_width = 0.002
        max_arrow_width = lw
        max_collision = np.sqrt(union_size)
        thick_a = (max_arrow_width - min_arrow_width)/(1.*max_collision - 1)
        thick_b = max_arrow_width - 1.*max_collision*thick_a
        for j in range(0, size - 1):
            a, b = pairs[2 * j], pairs[2 * j + 1]
            start, end = pos[a], pos[b]

            i = next_label - union_size
            aa, bb = self.fast_find(uf, a), self.fast_find(uf, b)

            a = (uf[a] != 0) * (aa - union_size)
            b = (uf[b] != 0) * (bb - union_size)
            uf[aa] = uf[bb] = next_label
            next_label += 1

            na, nb = sz[a], sz[b]
            sz[i] = na + nb

            size_reflection = np.sqrt(min(na, nb))
            w = size_reflection*thick_a + thick_b
            arr = Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], color = default_color, width=w)
            ax.add_patch(arr)

        # line_collection.set_array(self._mst[:, 2].T)

    def plot(self, axis=None, node_size=40, node_color=None,
             node_alpha=0.8, edge_alpha=0.15, edge_linewidth=8, vary_line_width=True,
             core_color = 'purple'):
        """Plot the minimum spanning tree (as projected into 2D by t-SNE if required).

        Parameters
        ----------

        axis : matplotlib axis, optional
               The axis to render the plot to

        node_size : int, optional (default 40)
                The size of nodes in the plot.

        node_color : matplotlib color spec, optional
                By default draws colors according to labels
                where alpha regulated by cluster size.

        node_alpha : float, optional (default 0.8)
                The alpha value (between 0 and 1) to render nodes with.

        edge_alpha : float, optional (default 0.4)
                The alpha value (between 0 and 1) to render nodes with.

        edge_linewidth : float, optional (default 2)
                The linewidth to use for rendering edges.

        vary_line_width : bool, optional (default True)
                By default, edge thickness and color depends on the size of
                the clusters connected by it.
                Thicker edge connects a bigger clusters.
                Red color indicates emergence of the cluster.

        core_color : matplotlib color spec, optional (default 'purple')
                Plots colors at the node centers.
                Can be omitted by passing None.

        Returns
        -------

        axis : matplotlib axis
                The axis used the render the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        if self._data.shape[0] > 32767:
            warn('Too many data points for safe rendering of a minimal spanning tree!')
            return None

        if axis is None:
            axis = plt.gca()
            axis.set_axis_off()

        pos = self.decrease_dimensions()

        if node_color is None:
            axis.scatter(pos.T[0], pos.T[1], c=self.get_node_colors(self._labels), s=node_size, alpha=node_alpha)
        else:
            axis.scatter(pos.T[0], pos.T[1], c=node_color, s=node_size)

        if vary_line_width:
            self.draw_druhg_edges(axis, self._mst_pairs, pos, edge_linewidth, edge_alpha)
        else:
            self.draw_simple_edges(axis, self._mst_pairs, pos, edge_linewidth, edge_alpha)

        # axis.set_xticks([])
        # axis.set_yticks([])

        if core_color is not None:
            # adding dots at the node centers
            axis.scatter(pos.T[0], pos.T[1], c=core_color, marker='.', s=node_size / 10)

        return axis

    def to_numpy(self):
        """Return a numpy array of pairs of from and to in the minimum spanning tree
        """
        return self._mst_pairs.copy()

    def to_pandas(self):
        """Return a Pandas dataframe of the minimum spanning tree.

        Each row is an edge in the tree; the columns are `from` and `to`
        which are indices into the dataset
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError('You must have pandas installed to export pandas DataFrames')

        result = DataFrame({'from': self._mst_pairs[::2].astype(int),
                            'to': self._mst_pairs[1::2].astype(int),
                            'distance': None})
        return result

    def to_networkx(self):
        """Return a NetworkX Graph object representing the minimum spanning tree.

        Nodes have a `data` attribute attached giving the data vector of the
        associated point.
        """
        try:
            from networkx import Graph, set_node_attributes
        except ImportError:
            raise ImportError('You must have networkx installed to export networkx graphs')

        result = Graph()
        size = int(len(self._mst_pairs)/2)
        for i in range(0, size):
            result.add_edge(self._mst_pairs[2*i], self._mst_pairs[2*i+1])

        data_dict = {index: tuple(row) for index, row in enumerate(self._data)}
        set_node_attributes(result, data_dict, 'data')

        return result
