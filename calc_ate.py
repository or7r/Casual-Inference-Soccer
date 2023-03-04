import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd 

def fit_model(data, out_var, ignore=None, max_iter=10000, model_type="logistic"):

    if ignore is None:
        ignore = []
    
    if model_type == "logistic":
        model = LogisticRegression(max_iter=max_iter)
    elif model_type == "linear":
        model = LinearRegression()
    elif isinstance(model_type, sklearn.base.BaseEstimator):
        model = model_type
    else:
        raise ValueError("model_type must be 'logistic' or 'linear'")

    X = data.drop([out_var] + ignore, axis=1)
    y = data[out_var]

    model.fit(X, y)

    return model

def propensity_fitting(data):

    propensity_func = fit_model(data, "T", ignore=["Y"], max_iter=10000)

    return propensity_func


def calc_propensity_score(data, propensity_func):
    X = data.drop(["T", "Y"], axis=1)
    propensity_score = propensity_func.predict_proba(X)[:, 1]
    propensity_score = pd.Series(propensity_score, index=X.index)

    return propensity_score

def IPW_ATE(Y, T, propensity_score):
    return (np.sum(Y[T == 1] / propensity_score[T == 1]) - np.sum(Y[T == 0] / propensity_score[T == 0])) / len(Y)

def calculate_IPW_ATE(data):
    propensity_func = propensity_fitting(data)
    propensity_score = calc_propensity_score(data, propensity_func)
    return IPW_ATE(data["Y"], data["T"], propensity_score)


# S-learner
def fit_y(data):

    y_func = fit_model(data, "Y", max_iter=10000, model_type="linear")

    return y_func

def covariate_adjustment(data, y1_func, y0_func=None):
    if y0_func is None:
        y0_func = y1_func

    X = data.drop(["Y"], axis=1)

    X1 = X.copy()
    X1["T"] = 1
    y1 = y1_func.predict(X1)

    X0 = X.copy()
    X0["T"] = 0
    y0 = y0_func.predict(X0)

    y = y1 - y0

    return y.mean()




def calculate_S_learner_ATE(data):

    y_func = fit_y(data)

    return covariate_adjustment(data, y_func)

# T-learner
def calculate_T_learner_ATE(data):

    y1_func = fit_y(data[data["T"] == 1])
    y0_func = fit_y(data[data["T"] == 0])

    return covariate_adjustment(data, y1_func, y0_func)

from sklearn.neighbors import NearestNeighbors
def calculate_matching_ATE(data, metric="euclidean"):

    knn0 = NearestNeighbors(n_neighbors=1, metric=metric)
    knn1 = NearestNeighbors(n_neighbors=1, metric=metric)

    X = data.drop(["Y", "T"], axis=1)

    knn0.fit(X[data["T"] == 0])
    knn1.fit(X[data["T"] == 1])

    X0 = X[data["T"] == 0]
    X1 = X[data["T"] == 1]

    _, idx0 = knn0.kneighbors(X1)
    _, idx1 = knn1.kneighbors(X0)

    y0 = data["Y"][data["T"] == 0].iloc[idx0[:, 0]]
    y1 = data["Y"][data["T"] == 1].iloc[idx1[:, 0]]

    y0 = data["Y"][data["T"] == 1] - y0
    y1 = y1 - data["Y"][data["T"] == 0]

    return (y0.sum() + y1.sum()) / (len(y0) + len(y1)) 




def calculate_ATE(data, method="IPW"):
    if method == "IPW":
        return calculate_IPW_ATE(data)
    elif method == "S-learner":
        return calculate_S_learner_ATE(data)
    elif method == "T-learner":
        return calculate_T_learner_ATE(data)
    elif method == "matching":
        return calculate_matching_ATE(data)
    else:
        raise ValueError("method must be 'IPW', 'S-learner', 'T-learner', or 'matching'")

