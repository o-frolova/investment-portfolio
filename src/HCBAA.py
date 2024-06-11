import numpy as np
import pandas as pd
from Stocks import *
import math 
from OptimalCluster.opticlust import Optimal
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import riskfolio as rp
from scipy.stats import kendalltau
from Hoeffding import *

# HCBAA - Hierarchical Clustering-Based Asset Allocation
class HCBAA():
    def __init__(self, Stocks: list, tickers: list) -> None:
        """
        Constructor for the HCBAA (Hierarchical Clustering-Based Asset Allocation) class.
        Parameters:
        - Stocks (list): List of stock objects.
        - tickers (list): List of stock tickers.
        """
        self.__Stocks = Stocks
        self.__tickers = tickers
        self.__correlation_matrix = self.__get_correlation_matrix()
        self.__distance_matrix = self.__get_distance_matrix()
        self.__optimal_number_clusters = self.__get_optimal_number_clusters()
        self.__clustering_methods = ['single', 'complete', 'average', 'ward', 'DBHT']
        self.__covariance_matrix = self.__get_covariance_matrix()
    
    @property
    def optimal_number_clusters(self,) -> int:
        return self.__optimal_number_clusters
        
    @property
    def distance_matrix(self,) -> pd.DataFrame:
        return self.__distance_matrix
    
    @property
    def correlation_matrix(self,) -> pd.DataFrame:
        return self.__correlation_matrix
    
    @property
    def covariance_matrix(self,) -> np.array:
        return self.__covariance_matrix

    def __get_covariance_matrix(self,) -> pd.DataFrame:
        """
        Calculates the covariance matrix for the given list of stock objects.
        Returns:
        - pd.DataFrame: Covariance matrix.
        """
        covariance_matrix = []
        for stock_1 in self.__Stocks:
            covv = []
            for stock_2 in self.__Stocks:
                covv.append(np.cov(stock_1.returns, stock_2.returns)[0, 1])
            covariance_matrix.append(covv)

        return pd.DataFrame(covariance_matrix, columns = self.__tickers, index = self.__tickers)

    def __get_correlation_matrix(self,) -> pd.DataFrame:
        """
        Calculate the correlation matrix between stocks.(Pearson) 
        Returns:
        - pd.DataFrame: Correlation matrix.
        """
        correlation_matrix = []
        for stock_1 in self.__Stocks:
            correlation_row = []
            for stock_2 in self.__Stocks:

                # tau, p_value = kendalltau(stock_1.returns, stock_2.returns)
                # correlation_row.append(tau)

                # correlation_row.append(hoeffding(np.array(stock_1.returns), np.array(stock_2.returns)))

                correlation_row.append(np.corrcoef(stock_1.returns, stock_2.returns)[0, 1])
            correlation_matrix.append(correlation_row)

        return pd.DataFrame(correlation_matrix, columns = self.__tickers, index = self.__tickers)
    
    def __get_distance_matrix(self,) -> pd.DataFrame:
        """
        Calculate the distance matrix based on correlation matrix.
        Returns:
        - pd.DataFrame: Distance matrix.
        """
        distance_matrix = []
        for stock_1 in self.__Stocks:
            distance_row = []
            for stock_2 in self.__Stocks:
                distance_row.append(round(math.sqrt(2*(1-self.__correlation_matrix[stock_1.ticker][stock_2.ticker])), 5))
            distance_matrix.append(distance_row)
        return pd.DataFrame(distance_matrix, columns = self.__tickers, index = self.__tickers)
    
    def __get_optimal_number_clusters(self, ) -> int:
        """
        Determine the optimal number of clusters.
        Returns:
        - int: Optimal number of clusters.
        """
        returns_stocks = pd.DataFrame()

        returns_stocks = pd.concat([stock.returns for stock in self.__Stocks], axis=1)
        returns_stocks.columns = [stock.ticker for stock in self.__Stocks]
        optimal_k = Optimal()
        optimal_num_clusterts = optimal_k.gap_stat_se(returns_stocks.T, upper = len(self.__Stocks), )

        print('Optimum clusters: ', optimal_num_clusterts)
        return optimal_num_clusterts
    

    def __cluster_allocation(self, linkage_matrix: list, clustering_method: str) -> dict:
        """
        Perform cluster allocation based on the hierarchical clustering linkage matrix.
        Parameters:
        - linkage_matrix (list): Hierarchical clustering linkage matrix.
        - clustering_method (str): Method used for hierarchical clustering.
        Returns:
        - dict: History of cluster allocations.
        """
        history = {}
        step = 0
        old_groups_clusters = []
        for i in range(len(self.__distance_matrix)):
            old_groups_clusters.append({i: [i], 'weight': None})
        num_cl = len(self.__distance_matrix)
        history[step] = old_groups_clusters

        for line in linkage_matrix:

                step += 1
                first_index_cl = int(line[0])
                second_index_cl = int(line[1])

                new_group_clusters = []
                for cluster in old_groups_clusters:

                    if list(cluster.keys())[0] != first_index_cl and list(cluster.keys())[0] != second_index_cl:
                        new_group_clusters.append({list(cluster.keys())[0]: list(cluster.values())[0], 'weight': None})
                    if list(cluster.keys())[0] == first_index_cl:
                        first_cl = cluster
                    if list(cluster.keys())[0] == second_index_cl:
                        second_cl = cluster

                new_group_clusters.append({num_cl: list(first_cl.values())[0] + list(second_cl.values())[0], 'weight': None })

                num_cl += 1
                old_groups_clusters = new_group_clusters
                history[step] = new_group_clusters

        return history
    

    def __get_history_clustering(self, clustering_method: str) -> dict:
        """
        Get the history of cluster allocations for a specific clustering method.
        Parameters:
        - clustering_method (str): Method used for hierarchical clustering.
        Returns:
        - dict: History of cluster allocations.
        """
        arr = self.__distance_matrix.to_numpy()
        np.fill_diagonal(arr, 0)
        self.__distance_matrix = pd.DataFrame(arr, index=self.__distance_matrix.index, columns=self.__distance_matrix.columns)
        
        if clustering_method == 'DBHT':
            distance_matrix = []
            new_correlation_matrix = self.__correlation_matrix + 1

            for stock_1 in self.__Stocks:
                distance_row = []
                for stock_2 in self.__Stocks:
                    distance_row.append(round(math.sqrt(2*(2-new_correlation_matrix[stock_1.ticker][stock_2.ticker])), 5))
                distance_matrix.append(distance_row)
            new_distance = pd.DataFrame(distance_matrix, columns = self.__tickers, index = self.__tickers)
            
            arr = new_distance.to_numpy()
            np.fill_diagonal(arr, 0)
            new_distance = pd.DataFrame(arr, index=new_distance.index, columns=new_distance.columns)
            _, _, _, _, _, linkage_matrix = rp.DBHT.DBHTs(D = np.array(new_distance), S = np.array(new_correlation_matrix))
        else:

            condensed_distance_matrix = squareform(self.__distance_matrix)
            linkage_matrix = linkage(condensed_distance_matrix, clustering_method)

        history = self.__cluster_allocation(linkage_matrix, clustering_method)
        return history
    
    def __distribution_weights_clusters(self, history: dict) -> list:
        """
        Distribute weights to clusters based on the clustering history.
        Parameters:
        - history (dict): History of cluster allocations.
        Returns:
        - list: Final distribution of weights to clusters.
        """
        history[list(history.keys())[-1]][0]['weight'] = 1
        for step_index in list(history.keys())[::-1][:-1]:
            cluster_dubl = []
            for cluster_1 in history[step_index]:
                for cluster_2 in history[step_index - 1]:
                    if list(cluster_1.keys())[0] == list(cluster_2.keys())[0]:
                        cluster_dubl.append(list(cluster_1.keys())[0])
                        cluster_2['weight'] = cluster_1['weight']

            for cluster in history[step_index - 1]:
                if cluster['weight'] == None:
                    weight = 0
                    for cluster_ in history[step_index]:
                        if list(cluster_.keys())[0] not in cluster_dubl:
                            weight = cluster_['weight'] / 2
                    cluster['weight'] = weight
        
        for step_index in list(history.keys()):
            if len(history[step_index]) == self.__optimal_number_clusters:
                return history[step_index]
    
    def __distribution_weights_assets(self, weight_clusters: list, clustering_method: str) -> None:
        """
        Distribute weights to individual assets based on cluster weights.
        Parameters:
        - weight_clusters (list): List of cluster weights.
        - clustering_method (str): Method used for hierarchical clustering.
        """
        for cluster in weight_clusters:
            weight_for_each_asset = cluster['weight']/len(cluster[list(cluster.keys())[0]])
            for stock_index in cluster[list(cluster.keys())[0]]:
                self.__Stocks[stock_index].weight_cluster[clustering_method] = weight_for_each_asset
    
    def __asset_allocation_weights(self, clustering_method: str) -> None:
        """
        Perform asset allocation based on hierarchical clustering.
        Parameters:
        - clustering_method (str): Method used for hierarchical clustering.
        """
        history = self.__get_history_clustering(clustering_method)
        weight_clusters = self.__distribution_weights_clusters(history)

        self.__distribution_weights_assets(weight_clusters, clustering_method)
    
    def __get_portfolios_clustering(self,) -> dict:
        """
        Get portfolios for each clustering method.
        Returns:
        - dict: Portfolios for each clustering method.
        """
        portfolios_on_clustering = {}
        SL, CM, AV, WR, DBHT = [], [], [], [], []
        for stock in self.__Stocks:
            SL.append(stock.weight_cluster['single'])
            CM.append(stock.weight_cluster['complete'])
            AV.append(stock.weight_cluster['average'])
            WR.append(stock.weight_cluster['ward'])
            DBHT.append(stock.weight_cluster['DBHT'])

        portfolios_on_clustering['single'] = SL
        portfolios_on_clustering['complete'] = CM
        portfolios_on_clustering['average'] = AV
        portfolios_on_clustering['ward'] = WR
        portfolios_on_clustering['DBHT'] = DBHT
        return portfolios_on_clustering
    
    def HCBAA_portfolios(self,):
        """
        Perform Hierarchical Clustering-Based Asset Allocation (HCBAA).
        Returns:
        - dict: Portfolios for each clustering method.
        """
        for clustering_method in self.__clustering_methods:
            self.__asset_allocation_weights(clustering_method)
        portfolios_on_clustering = self.__get_portfolios_clustering()
        return portfolios_on_clustering
