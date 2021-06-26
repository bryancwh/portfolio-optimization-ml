#%%
# Import python packages
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

import main_functions.return_models as re_m
import main_functions.risk_models as ri_m
import main_functions.optimisation_function as of
import main_functions.import_datasets as imp_df

def main():
    tickers = ['AC.TO','ZSP.TO','XFN.TO','HEU.TO','XIT.TO']
    date_from = pd.to_datetime('2013-01-01')
    date_to = pd.to_datetime('2020-06-13')

    df = imp_df.gen_df(tickers, date_from, date_to)

    return of.optimisation(df)

main()
# def main(df_stocks, risk_free=0, num_portfolios=1000):
#     all_returns = []
#     all_risks = []
#     sharpe_ratios = []
#     all_weights = []
#     n = len(df_stocks.columns)

#     for portfolio in range(num_portfolios):
#         # Generate random portfolio weights
#         weight = np.random.random_sample(n)
#         weight = np.round((weight / np.sum(weight)), 3)
#         all_weights.append(weight)

#         # Calculate returns
#         returns_matrix = return_models.cagr_matrix(df_stocks)
#         port_returns = return_models.portfolio_returns(returns_matrix, weight)
#         all_returns.append(port_returns)

#         # Caclulate risks
#         cov_matrix = risk_models.covariance_matrix(df_stocks)
#         port_risks = risk_models.portfolio_volatility(cov_matrix, weight)
#         all_risks.append(port_risks)

#         # Calculate peformance metrics (Sharpe Ratio)
#         sharpe_ratio = (port_returns - risk_free)/port_risks
#         sharpe_ratios.append(sharpe_ratio)

#     portfolios_metrics = [all_returns, all_risks, sharpe_ratios, all_weights]

#     portfolios_df = pd.DataFrame(portfolios_metrics).transpose()

#     portfolios_df.columns = ['Return', 'Risk', 'Sharpe', 'Weights']

#     return portfolios_df
# %%
