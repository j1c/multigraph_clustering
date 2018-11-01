import graspy as gs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mgcpy.independence_tests.mgc.mgc import MGC


class MultiGraphCluster():
    def __init__(self,
                 transform='zero-boost',
                 laplacian=None,
                 n_elbows=2,
                 kclusters=50,
                 gclusters=50,
                 random_state=None):
        """
        Parameters
        ----------
        transform : None, or string {log, zero-boost, simple-all, simple-nonzero}
            log :
                Plots the log of all nonzero numbers
            zero-boost :
                Pass to ranks method. preserves the edge weight for all 0s, but ranks 
                the other edges as if the ranks of all 0 edges has been assigned. 
            'simple-all': 
                Pass to ranks method. Assigns ranks to all non-zero edges, settling 
                ties using the average. Ranks are then scaled by 
                    .. math:: \frac{2 rank(non-zero edges)}{n^2 + 1}
                where n is the number of nodes
            'simple-nonzero':
                Pass to ranks method. Same as 'simple-all' but ranks are scaled by
                    .. math:: \frac{2 rank(non-zero edges)}{num_nonzero + 1}
        """
        self.transform = transform
        self.laplacian = laplacian
        self.n_elbows = n_elbows
        self.kclusters = kclusters
        self.gclusters = gclusters
        self.random_state = random_state

    def _process_graphs(self, graphs, transform=None, laplacian=None):
        X = [gs.utils.import_graph(i) for i in graphs]
        if transform is not None:
            X = [gs.utils.pass_to_ranks(i, method=transform) for i in X]
        if laplacian is not None:
            X = [gs.utils.to_laplace(i, form=laplacian) for i in X]
        return X

    def _plot_scatter(self, X, Y, xlablel, ylabel, title):
        with sns.plotting_context("talk", font_scale=1):
            fig = plt.figure(figsize=(10, 6))
            plot = sns.scatterplot(X, Y)
            plot.set(xlabel=xlablel, ylabel=ylabel, title=title)
        return plot

    def _run_mgc(self, X, Y):
        mgc = MGC(X, Y, None)
        pval, data = mgc.p_value()

        return pval, data

    def _run_omni(self, graphs):
        omni = gs.embed.OmnibusEmbed(n_elbows=self.n_elbows)
        Xhat = omni.fit_transform(graphs)

        return Xhat

    def _run_cmds(self, X):
        print("Running cMDS")
        cmds = gs.embed.ClassicalMDS()
        Dhat = cmds.fit_transform(X)
        dissimilarity = cmds.dissimilarity_matrix_

        return Dhat, dissimilarity

    def _run_gclust(self, X, Y):
        print("Running Gaussian Clustering with {} clusters".format(
            self.gclusters))
        gclust = gs.cluster.GaussianCluster(
            max_components=self.gclusters, random_state=self.random_state)
        gclust.fit(X, Y)

        # bic plot
        if self.transform is not None:
            title = "GClust o cMDS o Omni o ZG(2) o PTR"
        else:
            title = "GClust o cMDS o Omni o ZG(2)"
        bic_plot = self._plot_scatter(
            range(1, self.gclusters + 1), gclust.bic_, "Number of Clusters",
            "BIC", title)

        # ari plot
        ari_plot = self._plot_scatter(
            range(1, self.gclusters + 1), gclust.ari_, "Number of Clusters",
            "ARI", title)

        return gclust, bic_plot, ari_plot

    def _run_kclust(self, X, Y, labels):
        print("Running KMeans on vectorized graphs with {} clusters.".format(
            self.kclusters))
        idx = np.triu_indices(self.n_vertices)
        X_vec = np.vstack([g[idx] for g in X])
        Y_vec = np.vstack([g[idx] for g in Y])
        Z = np.hstack([X_vec, Y_vec])

        kclust = gs.cluster.KMeansCluster(
            max_clusters=self.kclusters, random_state=self.random_state)
        kclust.fit(Z, labels)

        # bic plot
        if self.transform is not None:
            title = "KMeans o vec(A) o PTR"
        else:
            title = "KMeans o vec(A)"
        silhouette_plot = self._plot_scatter(
            range(2, self.kclusters + 1), kclust.silhouette_,
            "Number of Clusters", "Silhouette Score", title)

        # ari plot
        ari_plot = self._plot_scatter(
            range(2, self.kclusters + 1), kclust.ari_, "Number of Clusters",
            "ARI", title)

        return kclust, silhouette_plot, ari_plot

    def run_experiment(self, X, Y, labels):
        assert len(X) == len(Y)
        assert len(X) == len(labels)

        X_graphs = self._process_graphs(X, self.transform, self.laplacian)
        Y_graphs = self._process_graphs(Y, self.transform, self.laplacian)

        self.X_graphs_ = X_graphs
        self.Y_graphs_ = Y_graphs

        for x, y in zip(X_graphs, Y_graphs):
            assert x.shape == y.shape

        # Metadata
        n_graphs = len(X)
        self.n_graphs = n_graphs
        n_vertices = X_graphs[0].shape[0]
        self.n_vertices = n_vertices

        # First run kmeans
        kclust, kclust_silhouette_plot, kclust_ari_plot = self._run_kclust(
            X_graphs, Y_graphs, labels)
        self.kclust_silhouette_plot_ = kclust_silhouette_plot
        self.kclust_ari_plot_ = kclust_ari_plot
        self.kclust_ari_ = kclust.ari_
        self.kclust_silhouette_ = kclust.silhouette_

        # Run omni
        print("Running graph embedding")
        Xhat = self._run_omni(X)
        Yhat = self._run_omni(Y)

        Zhat = np.hstack([Xhat, Yhat])
        Zhat = Zhat.reshape((n_graphs, n_vertices, -1))
        self.Zhat = Zhat

        Dhat, dissimilarity = self._run_cmds(Zhat)
        # Make class attributes
        self.dissimilarity_matrix_ = dissimilarity
        self.dhat_ = Dhat

        # Run mgc

        gclust, gclust_bic_plot, gclust_ari_plot = self._run_gclust(
            Dhat, labels)
        self.gclust_bic_plot_ = gclust_bic_plot
        self.gclust_ari_plot_ = gclust_ari_plot
        self.gclust_ari_ = gclust.ari_
        self.gclust_bic_ = gclust.bic_
