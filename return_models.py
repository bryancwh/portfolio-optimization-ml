# Functions for estimating the expected returns of the assets.

import pandas as pd
import numpy as np
# Test1

# Calculate returns given prices in type pd.DataFrame
def percentage_returns(price):
    return price.pct_change().dropna(how="all")


# Calculate annualised compounded returns
def cagr_matrix(price):
    returns = percentage_returns(price)
    return (1 + returns).prod() ** (1 / (returns.count()/252)) - 1


# Calculate total returns of portfolio
def portfolio_returns(returns_matrix, weight):
    return returns_matrix.dot(weight)
