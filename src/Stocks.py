import pandas as pd
import numpy as np
class Stocks():
    """A class representing stock data."""
    def __init__(self, id_stock: int, ticker: str, close_prices: list = None, volumes: list = None, dates: list = None) -> None:
        """Initialize a Stocks instance with specified attributes."""
        self.__id_ = id_stock
        self.__ticker = ticker
        self.__close_prices = close_prices if close_prices is not None else []
        self.__volumes = volumes if volumes is not None else []
        self.__dates = dates if dates is not None else []
        self.__returns = self.__get_returns()
        self.__weight_cluster = {'single': None,'complete': None, 'average': None, 'ward': None, 'DBHT': None}
        self.returns_temp = []

    @property
    def id_(self) -> int:
        """Get the stock ID."""
        return self.__id_

    @property
    def ticker(self) -> str:
        """Get the stock ticker."""
        return self.__ticker

    @property
    def close_prices(self) -> list:
        """Get the stock close prices."""
        return self.__close_prices

    @property
    def volumes(self) -> list:
        """Get the stock volumes."""
        return self.__volumes

    @property
    def dates(self) -> list:
        """Get the stock dates."""
        return self.__dates

    @property
    def returns(self) -> pd.Series:
        """Get the stock returns."""
        return self.__returns

    @property
    def weight_cluster(self) -> pd.Series:
        """Get the stock weight_cluster."""
        return self.__weight_cluster

    # В СТАТЬЕ БЕРУТ ОБЫЧНУЮ ДОХОДНОСТЬ, А НЕ ЛОГАРИФМИЧЕСКУЮ !
    def __get_returns(self) -> pd.Series:
        """Calculating the non-logarithmic return on an stock """
        if self.__close_prices:
            returns = pd.Series(self.__close_prices).pct_change()
            returns.iloc[0] = 0
            return np.log(1 + returns)
            # return returns
        else:
            return []

    @returns.setter
    def returns(self, returns_):
        """Set stock returns"""
        self.__returns = returns_

    @dates.setter
    def dates(self, dates_):
        """Set stock dates"""
        self.__dates = dates_

    @close_prices.setter
    def close_prices(self, close_prices_):
        """Set stock close_prices"""
        self.__close_prices = close_prices_

    @volumes.setter
    def volumes(self, volumes_):
        """Set stock volumes"""
        self.__volumes = volumes_

    @weight_cluster.setter
    def weight_cluster(self, weight_cluster_):
        """Set stock weight_cluster"""
        self.__weight_cluster = weight_cluster_
    
    @id_.setter
    def id_(self, new_id):
        self.__id_ = new_id