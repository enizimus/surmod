import numpy as np


def shuffle_data(
    X: np.ndarray, y: np.ndarray, seed: int = None
) -> (np.ndarray, np.ndarray):
    """Shuffles the input data and returns it.

    Args:
        X (np.ndarray): Input dataset nondependant variables.
        y (np.ndarray): Input dataset dependent variable.
        seed (int): Random seed. Defaults to None.

    Returns:
        X (np.ndarray), y (np.ndarray): Shuffeled input data.
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = X.shape[0]

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    X = X[indices, :]
    y = y[indices]

    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, p: float = 0.1, seed=None
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Splits the input data into a training and test set according
    to percentage given by p (p gives the amount of test data).

    Args:
        X (np.ndarray): Input dataset nondependant variables.
        y (np.ndarray): Input dataset dependent variable.
        p (float, optional): Percentage of test data. Defaults to 0.1.

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray): Split data.
    """
    num_samples = X.shape[0]
    n = int((1.0 - p) * num_samples)

    X, y = shuffle_data(X, y, seed)

    X_train = X[:n, :]
    X_test = X[n:, :]

    y_train = y[:n]
    y_test = y[n:]

    return X_train, X_test, y_train, y_test


def cross_validation(
    model: object, X: np.array, y: np.array, k: int, metric="rmse", seed=None
) -> np.ndarray:
    """Implementation of k-fold cross validation for the passed model and data.

    Args:
        model (object): Model that implements fit and predict methods.
        X (np.array): Training data input features.
        y (np.array): Training data dependant variable.
        k (int): How many folds to perform.
        metric (str, optional): Metric used to compute score of cross validation. Defaults to "rmse".
        seed ([type], optional): Random seed. Defaults to None.

    Returns:
        np.ndarray: Scores of the k-folds.
    """
    num_samples = X.shape[0]
    assert num_samples > k, "k must be smaller than number of samples!"

    scores = np.zeros((k,))
    score_fun = get_scoring_fun(metric)

    X, y = shuffle_data(X, y, seed)
    m = num_samples // k

    for ik in range(k):

        ind_test = np.arange(ik * m, min((ik + 1) * m, num_samples))
        ind_train = np.concatenate(
            (np.arange(0, ind_test[0]), np.arange(ind_test[-1] + 1, num_samples))
        )

        X_test = X[ind_test, :]
        X_train = X[ind_train, :]

        y_test = y[ind_test]
        y_train = y[ind_train]

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        scores[ik] = score_fun(y_test, y_hat)

    return scores


def grid_search_CV(
    model: object, X: np.array, y: np.array, cv_k: int, metric="rmse", seed=None
):
    """Performs a grid search on parameter intervals defined in the model and uses
    cross validation of evaluation fo these parameters.

    Args:
        model (object): Model that implements fit and predict methods.
        X (np.array): Training data input features.
        y (np.array): Training data dependant variable.
        cv_k (int): How many folds to perform.
        metric (str, optional): Metric used to compute score of cross validation. Defaults to "rmse".
        seed ([type], optional): Random seed. Defaults to None.

    Returns:
        (float): Parameters for optimal score.
    """
    params = model.get_params()
    param_scores = np.zeros((params.shape[0],))

    for ix, param in enumerate(params):

        model.set_params(param)
        scores = cross_validation(model, X, y, k=cv_k, metric=metric, seed=seed)

        param_scores[ix] = np.sum(scores)

    best_param_ind = param_scores.argmin()
    return params[best_param_ind]


def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))


def abse(y, y_hat):
    return np.sum(np.abs(y - y_hat))


def get_scoring_fun(metric="rmse"):
    scoring_functions = {
        "rmse": rmse,
        "abse": abse,
    }

    return scoring_functions[metric]
