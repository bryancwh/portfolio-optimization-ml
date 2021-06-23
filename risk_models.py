# Functions for estimating the covariance matrix given historical returns

import pandas as pd
import numpy as np
from return_models import percentage_returns


# Calculate portfolio variance
def covariance_matrix(close):
    returns = percentage_returns(close)

    # multiply by 252 total trading days to annualise
    return returns.cov() * 252


def portfolio_variance(cov_matrix, weight):
    transpose_weight = np.transpose(weight)
    return transpose_weight.dot(cov_matrix.dot(weight))


# Square root of variance
def portfolio_volatility(cov_matrix, weight):
    transpose_weight = np.transpose(weight)
    port_var = transpose_weight.dot(cov_matrix.dot(weight))
    return np.sqrt(port_var)
