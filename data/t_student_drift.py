import numpy as np
from config import T, df, M
def generate_timeseries(
    M=M,
    T=T,
    df=df,
):
    """
    Input:
    M : nombre de séries temporelles
    T : longueur des séries temporelles
    df : degré de liberté de la loi de Student

    Génère M séries temporelles de longeur T xi,t selon une loi de Student avec
    df = 4, de variance 1 et de moyenne moyenne µk pour 100 valeurs de µk entre
    -2 et +2 (100*M séries temporelles de longueur T en tout).

    """
    mu = np.linspace(-2, 2, 100)  # 100 valeurs de mu entre -2 et 2
    # M séries temporelles de longueur T pour chaque mu ;
    # e.g xi[i, :, j] correspond à la ième série temporelle de longueur T avec la jème moyenne mu[j]
    xi = np.zeros((M, T, len(mu)))
    for i in range(M):
        for j in range(len(mu)):
            xi[i, :, j] = np.random.standard_t(df, T) + mu[j]

    return xi, mu
