# -*- coding: utf-8 -*-
# Author: Pavel "DRUHG" Artamonov
# License: 3-clause BSD


import numpy as np

from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from warnings import warn

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
        else:
            projection = self._data.copy()
        #     projection = dict(enumerate(projection))
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

        size = len(pairs) / 2
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

    def draw_druhg_edges(self, ax, pairs, pos, lw=1., alpha=0.7):
        try:
            from matplotlib import collections as mc
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        size = len(pairs) / 2 + 1
        full_size = size
        uf, sz, ms = np.zeros(2 * size, dtype=int), np.ones(size), np.zeros(size)
        next_label = size + 1

        lines, line_widths, line_colors = [], [], []
        heads, heads_widths, heads_colors = [], [], []
        default_color, cluster_color = (0, 0, 0), (1., 0., 0.)
        head_scale, head_alpha = 4., 0.8 * alpha
        for j in range(0, size - 1):
            a, b = pairs[2 * j], pairs[2 * j + 1]
            start, end = pos[a], pos[b]

            hh = []
            i = next_label - full_size
            aa, bb = self.fast_find(uf, a), self.fast_find(uf, b)

            a = (uf[a] != 0) * (aa - full_size)
            b = (uf[b] != 0) * (bb - full_size)
            uf[aa] = uf[bb] = next_label
            next_label += 1

            na, nb = sz[a], sz[b]
            sz[i] = na + nb
            size_reflection = np.sqrt(min(na, nb))
            dip = 1. / np.sqrt(min(na, nb))
            # ----------------------
            new_mass = 0.
            old_mass = ms[a]
            excess_of_mass = 1. - dip - old_mass
            if excess_of_mass > old_mass:
                new_mass = excess_of_mass
                hh.append(0)
            else:
                new_mass = old_mass
            # ----------------------
            old_mass = ms[b]
            excess_of_mass = 1. - dip - old_mass
            if excess_of_mass > old_mass:
                new_mass = self.merge_means(na, new_mass, nb, excess_of_mass)
                hh.append(1)
            else:
                new_mass = self.merge_means(na, new_mass, nb, old_mass)
            # ----------------------
            ms[i] = new_mass

            # visualisation
            hw = min(size_reflection * lw * head_scale, 1.5 * lw * size_reflection)
            if (na == 1 and nb == 1) or len(hh) != 1:  # double cluster reflection or ...
                col = cluster_color
                if len(hh) == 0:  # outliers
                    col = default_color
                if na == 1 and nb == 1:  # double headed. Pure reciprocity
                    heads.append([start, ((start + end) / 2 + start) / 2])
                    heads_widths.append(hw)
                    heads_colors.append(col + (head_alpha,))

                lines.append([start, end])
                line_widths.append(size_reflection * lw)
                line_colors.append(col + (alpha,))
                heads.append([((start + end) / 2 + end) / 2, end])
                heads_widths.append(hw)
                heads_colors.append(col + (head_alpha,))
            else:  # there are two halves of different colors
                col = default_color
                if hh[0] == 0:
                    col = cluster_color
                lines.append([start, (start + end) / 2])
                line_widths.append(size_reflection * lw)
                line_colors.append(col + (alpha,))
                # the other half
                if col == cluster_color:
                    col = default_color
                else:
                    col = cluster_color
                lines.append([(start + end) / 2, end])
                line_widths.append(size_reflection * lw)
                line_colors.append(col + (alpha,))
                heads.append([((start + end) / 2 + end) / 2, end])
                heads_widths.append(hw)
                heads_colors.append(col + (head_alpha,))

        lc = mc.LineCollection(lines, colors=line_colors, linewidths=line_widths)
        ax.add_collection(lc)

        lc = mc.LineCollection(heads, colors=heads_colors, linewidths=heads_widths)
        ax.add_collection(lc)

        # line_collection.set_array(self._mst[:, 2].T)

    def plot(self, axis=None, node_size=40, node_color=None,
             node_alpha=0.8, edge_alpha = 0.8, edge_linewidth=2, vary_line_width=True,
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

        edge_alpha : float, optional (default 0.8)
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
            warn('Too many data points for safe rendering of an minimal spanning tree!')
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
        size = len(self._mst_pairs)/2
        for i in range(0, size):
            result.add_edge(self._mst_pairs[2*i], self._mst_pairs[2*i+1])

        data_dict = {index: tuple(row) for index, row in enumerate(self._data)}
        set_node_attributes(result, data_dict, 'data')

        return result
