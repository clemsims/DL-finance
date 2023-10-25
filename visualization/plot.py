import matplotlib.pyplot as plt
import numpy as np
from models.predict_model import plot_generalization_evaluation


def plot_sample(xi, mu, M):
    """
    Input:
    xi : séries temporelles générées par generate_timeseries
    mu : moyennes générées par generate_timeseries
    M : nombre de séries temporelles générées par generate_timeseries

    Affiche 5 séries temporelles générées par generate_timeseries avec 5 moyennes
    générées par generate_timeseries.
    """

    fig = plt.figure(figsize=(20, 20))
    # extract 5 random series with 5 random means
    idx = np.random.randint(0, M, 5)
    idx_mu = np.random.randint(0, len(mu), 5)

    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(xi[idx[i], :, idx_mu[i]])
        plt.title("Série temporelle #{} avec mu = {}".format(
            idx[i], np.round(mu[idx_mu[i]], 2)))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.grid(True)
    plt.show()
    return fig


def plot_ts_mu(xi, mu, M, idx_mu=40):
    """
    Plot toutes les séries temporelles pour un même mu superposées
    """
    fig = plt.figure(figsize=(20, 10))
    for i in range(M):
        plt.plot(xi[i, :, idx_mu])

    plt.title("Toutes les séries temporelles pour mu = {}".format(mu[idx_mu]))
    plt.xlabel("t")
    plt.ylabel("x")
    plt.grid(True)
    plt.show()
    return fig
