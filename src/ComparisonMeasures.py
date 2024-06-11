from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np 

class ComparisonMeasures():

    def __get_changing_weight(self, index: int, portfolio_returns: dict) -> list:
        """
        Private method to extract the weights of a specific index from portfolio returns.

        Parameters:
        - index (int): Index of the weight to extract.
        - portfolio_returns (dict): Dictionary containing portfolio returns.

        Returns:
        - list: List of weights for the specified index.
        """
        results_weights = [item[1][index] for item in portfolio_returns]
        return results_weights

    def __the_sharp_ratio_simple(self, returns: list) -> float:
        """
        Private method to calculate the simple Sharpe ratio.

        Parameters:
        - returns (list): List of returns.

        Returns:
        - float: Simple Sharpe ratio.
        """
        profi = sum(returns)/len(returns)
        return profi/(np.array(returns).std())

    def __adjusted_sharp_ratio(self, returns: list) -> float:
        """
        Private method to calculate the adjusted Sharpe ratio.

        Parameters:
        - returns (list): List of returns.

        Returns:
        - float: Adjusted Sharpe ratio.
        """
        mu_3 = skew(returns, axis=0, bias=True)
        mu_4 = kurtosis(returns, axis=0, bias=True)
        SR = self.__the_sharp_ratio_simple(returns)
        return SR*(1+(mu_3/6)*SR-((mu_4 - 3)/24)*pow(SR,2))

    def __certainty_equivalent_return(self, returns: list ) -> float:
        """
        Private method to calculate the certainty-equivalent return.

        Parameters:
        - returns (list): List of returns.

        Returns:
        - float: Certainty-equivalent return.
        """

        return 100 * (np.mean(returns) - 1/2 * pow(np.array(returns).std(), 2))
    
    def __the_max_drawdown(self, returns: list) -> float:
        """
        Private method to calculate the maximum drawdown.

        Parameters:
        - returns (list): List of returns.

        Returns:
        - float: Maximum drawdown.
        """
        max_price_portfolio = max(returns)
        min_price_portfolio = min(returns)

        return 100 * ((max_price_portfolio - min_price_portfolio)/ max_price_portfolio)
    


    def __to(self, portfolio_returns: dict) -> float:
        """
        Private method to calculate the turnover.

        Parameters:
        - portfolio_returns (dict): Dictionary containing portfolio returns.

        Returns:
        - float: Turnover percentage.
        """

        summary_weights_delta = 0
        portfolio_returns = portfolio_returns[::-1]
        for day_index in range(len(portfolio_returns) - 1):

            for weigth_index in range(len(portfolio_returns[0][1])):
                summary_weights_delta += np.abs(portfolio_returns[day_index][1][weigth_index] - portfolio_returns[day_index + 1][1][weigth_index])

        return 100 *  (summary_weights_delta/ len(portfolio_returns))

    def __sum_squared_portfolio_weights(self, portfolio_results: dict) -> float:
        """
        Private method to calculate the sum of squared portfolio weights.

        Parameters:
        - portfolio_results (dict): Dictionary containing portfolio results.

        Returns:
        - float: Sum of squared portfolio weights.
        """
        summary_weights = 0
        for day in portfolio_results:
            for weight in day[1]:
                summary_weights += pow(weight, 2)

        return summary_weights/len(portfolio_results)
    

    def get_metric(self, portfolio_results: dict) -> dict:
        """
        Public method to calculate various metrics for the portfolio.

        Parameters:
        - portfolio_results (dict): Dictionary containing portfolio results.

        Returns:
        - dict: Dictionary containing calculated metrics.
        """
        result = {}
        result['SR'] = round(self.__the_sharp_ratio_simple([t[0] for t in portfolio_results]),3)
        result['ASR'] = round(self.__adjusted_sharp_ratio([t[0] for t in portfolio_results]),3)
        result['CEQ(%)'] = round(self.__certainty_equivalent_return([t[0] for t in portfolio_results]),3)
        result['TO(%)'] = round(self.__to(portfolio_results), 3)
        result['MD(%)'] = round(self.__the_max_drawdown([t[0] for t in portfolio_results]),3)
        result['SSPW'] = round(self.__sum_squared_portfolio_weights(portfolio_results),3)
        return result
        