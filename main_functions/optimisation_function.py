#%%
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import return_models as re_m
import risk_models as ri_m

def optimisation(df_stocks):

    tickers = list(df_stocks.columns)
    num_tickers = len(tickers)

    # Set objective function
    def objective(weights_in):
        '''
        Objective function to maximize Sharpe Ratio
        '''
        # Calculate returns
        returns_matrix = re_m.cagr_matrix(df_stocks)
        port_returns = re_m.portfolio_returns(returns_matrix, weights_in)

        # Calculate risks
        cov_matrix = ri_m.covariance_matrix(df_stocks)
        port_risks = ri_m.portfolio_volatility(cov_matrix, weights_in)

        # Calculate performance metrics (Sharpe Ratio)
        sharpe_ratio = port_returns /port_risks

        # Return Sharpe Ratio 
        return -sharpe_ratio # Negative because optimize can only return minimum

    # Set constraints
    def constraint1(x):
        '''
        Constraint where all weights add up to 100
        '''
        return sum(x) - 100

    constraint = {
        'type': 'eq',
        'fun': constraint1
    }

    # Set bounds
    indiv_bounds = (0,100)
    bounds = tuple(indiv_bounds for x in range(num_tickers))

    # Get solution
    solution = minimize(
        objective,
        np.array([0.1 for tick in tickers]), 
        method  = 'SLSQP', 
        bounds = bounds,
        constraints = [constraint]
    )
        
    print(solution.x)
    print(sum(solution.x))

    # Return optimum solution
    return solution.x
# %%
