import numpy as np 
import pandas as pd
from Stocks import *
import riskparityportfolio as rpf
import math

# https://arxiv.org/abs/1311.4057
class CCD():
    def __init__(self, Stocks: list, tickers: list) -> None:
        """
        Constructor for the CCD class.
        Parameters:
        - Stocks: List of stock objects.
        - tickers: List of stock tickers.
        """
        self.__covergence_tol = 1e-10
        self.__max_iter = 10000
        self.__Stocks = Stocks
        self.__tickers = tickers
        self.__covariance_matrix = self.__get_covariance_matrix()
        self.__types_beta = ['MV', 'MDP', 'ERC', 'IVRB']
    
    @property
    def covergence_tol(self,) -> float:
        return self.__covergence_tol

    @property
    def max_iter(self,) -> int:
        return self.__max_iter
    
    @covergence_tol.setter
    def covergence_tol(self, new_covergence_tol: float) -> None:
        self.__covergence_tol = new_covergence_tol
    
    @max_iter.setter
    def max_iter(self, new_max_iter: int) -> None:
        self.__max_iter = new_max_iter
    
    @property
    def covariance_matrix(self,) -> np.array:
        return self.__covariance_matrix
    
    def __get_covariance_matrix(self,) -> np.array:
        """
        Calculates the covariance matrix for the given list of stock objects.
        Returns:
        - np.array: Covariance matrix.
        """
        covariance_matrix = []
        for stock_1 in self.__Stocks:
            covv = []
            for stock_2 in self.__Stocks:
                covv.append(np.cov(stock_1.returns, stock_2.returns)[0, 1])
            covariance_matrix.append(covv)
        return  np.array(covariance_matrix)
    
    def __beta(self, type_beta: str, i: int, weights: list) -> float:
        """
        Calculates the beta value for a specific stock in the given portfolio.
        Parameters:
        - type_beta: Type of beta calculation.
        - i: Index of the stock.
        - weights: Current portfolio weights.
        - Sx: Portfolio standard deviation.
        Returns:
        - float: Beta value.
        """
        if type_beta == 'MV':
            return weights[i]

        if type_beta == 'MDP':
            part1 = weights[i] * math.sqrt(self.__covariance_matrix[i][i])
            summary = 0
            for j in range(len(weights)):
                summary += weights[j] * math.sqrt(self.__covariance_matrix[j][j])
            return part1/summary

        if type_beta == 'ERC':
            return 1/len(weights)

        if type_beta == 'IVRB':
            part1 = math.pow(self.__covariance_matrix[i][i], -1)
            summary = 0
            for j in range(len(weights)):
                summary += math.pow(self.__covariance_matrix[j][j], -1)
            return part1/summary
    
    def __get_betas(self, weights: list, type_beta: str) -> list:
        betas = []
        for i in range(len(weights)):
            betas.append(self.__beta(type_beta, i, weights))
        return betas 
    
    def __check_beta(self, type_beta, weights) -> None:
        """
        Check the beta coefficients for non-negativity and their sum equal to one.
        Parameters:
        - type_beta: str, the type of beta coefficient
        - weights: list, list of weights for assets
        Raises:
        - ValueError: If a negative beta value is encountered or the sum of beta coefficients is not equal to one.
        """
        summary = 0
        for i in range(len(weights)):
            beta_value = self.__beta(type_beta, i,  weights)
            if beta_value < 0:
                raise ValueError(f'Negative beta value encountered for {type_beta} and weight index {i}.')
            summary += self.__beta(type_beta, i,  weights)
        if round(summary, 1) != 1:
            raise ValueError(f'The sum of the beta coefficient is not equal to one for {type_beta}.')

    def __check_weights(self, weights: list) -> None:
            """
            Check the weights for non-negativity and their sum equal to one.

            Parameters:
            - weights: list, list of weights for assets

            Raises:
            - ValueError: If a negative weight value is encountered or the sum of weights is not equal to one.
            """
            summary = 0
            for weight in weights:
                if weight < 0:
                    raise ValueError('Negative weight value encountered.')
                summary += weight
            if round(summary, 1) != 1:
                raise ValueError(f'The sum of the weights is not equal to one.')

    def __CDD(self, type_beta: str) -> list:

        current_weights = 1 / np.diag(self.__covariance_matrix) ** 0.5 / (np.sum(1 / np.diag(self.__covariance_matrix) ** 0.5))
        previous_weigths = current_weights / 100
        current_betas = self.__get_betas(current_weights, type_beta)

        iters = 0
        cvg = False
        while not cvg:

            current_weights = rpf.vanilla.design(self.__covariance_matrix, current_betas)
            cvg = np.linalg.norm(current_weights - previous_weigths) <= self.__covergence_tol

            current_betas = self.__get_betas(current_weights, type_beta)

            previous_weigths = current_weights.copy()
            iters = iters + 1
            if iters >= 10_000:
                print("Maximum iteration reached during the CCD descent: {}".format(10_000))
                break
        return current_betas
    
    def RiskBudgetingPortfolios(self,) -> dict:
        """
        Generates risk budgeting portfolios for different types of beta.
        Returns:
        - dict: Results portfolio for each type of beta.
        """
        results_portfolio = {}
        for type_beta in self.__types_beta:
            results_portfolio[type_beta] = self.__CDD(type_beta)
            self.__check_beta(type_beta, results_portfolio[type_beta])
            self.__check_weights(results_portfolio[type_beta])
        return results_portfolio

    
