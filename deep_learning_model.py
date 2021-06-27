import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import return_models
import risk_models
import numpy as np

"""
LSTM - remembers past data, good for financial time series.
input_shape - 2 dimensional matrix.
Flatten neurons -  n neurons, 1 neuron for each ticker.
softmax - make the weights (neurons) sum to one. 
"""


# Sequential Model (Very convenient, not very flexible - 1 input 1 output)
def build_model(df_stocks, input_shape, outputs):

    model = Sequential(
        [
            LSTM(64, input_shape=input_shape),
            Flatten(),
            Dense(outputs, activation='softmax')
        ]
    )

    def sharpe_loss(_, y_pred):

        # Calculate returns
        returns_matrix = return_models.cagr_matrix(df_stocks)
        port_returns = return_models.portfolio_returns(returns_matrix, y_pred)

        # Caclulate risks
        cov_matrix = risk_models.covariance_matrix(df_stocks)
        port_risks = risk_models.portfolio_volatility(cov_matrix, y_pred)

        # Calculate peformance metrics (Sharpe Ratio)
        #sharpe_ratio = (port_returns - risk_free)/port_risks
        sharpe = port_returns/port_risks

        # since we want to maximize Sharpe, while gradient descent minimizes the loss,
        # we can negate Sharpe (the min of a negated function is its max)
        return -sharpe

    model.compile(
        loss=sharpe_loss,
        optimizer='adam'
    )
    return model


def get_allocations(df_stocks):
    '''
    Computes and returns the allocation ratios that optimize the Sharpe over the given data

    input: data - DataFrame of historical closing prices of various assets

    return: the allocations ratios for each of the given assets
    '''

    # data with returns
    data_w_ret = np.concatenate(
        [data.values[1:], data.pct_change().values[1:]], axis=1)

    data = data.iloc[1:]
    self.data = tf.cast(tf.constant(data), float)

    if self.model is None:
        self.model = self.__build_model(data_w_ret.shape, len(data.columns))

    fit_predict_data = data_w_ret[np.newaxis, :]
    self.model.fit(fit_predict_data, np.zeros(
        (1, len(data.columns))), epochs=20, shuffle=False)
    return self.model.predict(fit_predict_data)[0]
