import numpy as np
from config import T, df, M
import matplotlib.pyplot as plt
# logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_mu(model, mu_new=1.8):
    mu_new = 1.8
    xi_test = np.random.standard_t(df, T) + mu_new
    xi_test = xi_test.reshape(1, T)
    yi_test = model.predict(xi_test)

    logger.info("Prediction: {}".format(yi_test[0][0]))
    logger.info("True value: {}".format(mu_new))
    return mu_new, yi_test[0][0]


def eval_(model, N=10000, mu_test=1.8):
    xi_test = np.zeros((N, T))
    yi_test = np.zeros((N, 1))

    for i in range(N):
        xi_test[i, :] = np.random.standard_t(df, T) + mu_test
        yi_test[i, :] = mu_test

    assert xi_test.shape[0] == yi_test.shape[0]

    mse = model.evaluate(xi_test, yi_test, verbose=True)
    logger.info("MSE: {}".format(mse))
    return mse


def generalization_evaluation(model, N=10000):
    """
    Tries to estimate the performance of the model on other mu values (never encountered before)
    """
    mse = []
    mu_test = np.linspace(-10, 10, 20)

    for i in range(len(mu_test)):
        mse.append(eval_(model, N, mu_test[i]))
    return mu_test, mse


def plot_generalization_evaluation(mu_test, mse):
    plt.figure(figsize=(20, 5))
    plt.plot(mu_test, mse)
    plt.xlabel("mu")
    plt.ylabel("mse")
    plt.title(
        "MSE en fonction de mu pour un scope de 20 mu entre -10 et 10", fontsize=20)
    plt.grid(True)
    plt.show()
