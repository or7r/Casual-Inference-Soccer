import numpy as np
import pandas as pd 
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from copy import deepcopy


def fit_model(data, out_var, ignore=None, max_iter=10000, model_type="logistic", calc_error=False, **kwargs):

    if ignore is None:
        ignore = []
    
    if model_type == "logistic":
        model = LogisticRegression(max_iter=max_iter)
    elif model_type == "linear":
        model = LinearRegression()
    elif isinstance(model_type, sklearn.base.BaseEstimator):
        model = deepcopy(model_type)
    else:
        raise ValueError("model_type must be 'logistic' or 'linear'")

    X = data.drop([out_var] + ignore, axis=1)
    y = data[out_var]

    model.fit(X, y)

    if calc_error:
        test_data = kwargs.pop("test_data")
        test_X = test_data.drop([out_var] + ignore, axis=1)
        test_y = test_data[out_var]
        if model_type == "logistic":
            # log loss
            y_pred = model.predict_proba(test_X)[:, 1]
            error = sklearn.metrics.log_loss(test_y, y_pred)
        else:
            y_pred = model.predict(test_X)
            error = sklearn.metrics.mean_squared_error(test_y, y_pred)    
        return model, error

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
    return (np.sum(Y[T == 1] / propensity_score[T == 1]) - np.sum(Y[T == 0] / (1 - propensity_score[T == 0]))) / len(Y)

def calculate_IPW_ATE(data):
    propensity_func = propensity_fitting(data)
    propensity_score = calc_propensity_score(data, propensity_func)
    return IPW_ATE(data["Y"], data["T"], propensity_score)


# S-learner
def fit_y(data, model_type="linear", **kwargs):

    y_func = fit_model(data, "Y", max_iter=10000, model_type=model_type, **kwargs)

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




def calculate_S_learner_ATE(data, model_type="linear", **kwargs):


    if "test_data" in kwargs:
        test_data = kwargs.pop("test_data")
    else:
        test_data = data

    y_func = fit_y(data, model_type=model_type, test_data=test_data, **kwargs)

    if kwargs.get("calc_error"):
        y_func, error = y_func
        return covariate_adjustment(data, y_func), error

    return covariate_adjustment(data, y_func)

# T-learner
def calculate_T_learner_ATE(data, model_type="ridge", **kwargs):

    if model_type == "ridge":
        model_type = sklearn.linear_model.Ridge(alpha=1e-5, max_iter=10000)


    if "test_data" in kwargs:
        test_data = kwargs.pop("test_data")
    else:
        test_data = data

    y1_func = fit_y(data[data["T"] == 1], model_type=model_type, test_data=test_data[test_data["T"] == 1],**kwargs)
    y0_func = fit_y(data[data["T"] == 0], model_type=model_type, test_data=test_data[test_data["T"] == 0], **kwargs)

    if kwargs.get("calc_error"):
        y1_func, error1 = y1_func
        y0_func, error0 = y0_func

        error = (error1 * len(test_data[test_data["T"] == 1]) + error0 * len(test_data[test_data["T"] == 0])) / len(data)
        return covariate_adjustment(data, y1_func, y0_func), error

    return covariate_adjustment(data, y1_func, y0_func)

def calculate_matching_ATE(data, metric="euclidean"):


    knn0 = NearestNeighbors(n_neighbors=1, metric=metric)
    knn1 = NearestNeighbors(n_neighbors=1, metric=metric)

    X = data.drop(["Y", "T"], axis=1)
    X = StandardScaler().fit_transform(X) # normalize data before matching

    x0_mask = data["T"] == 0
    x1_mask = data["T"] == 1

    X0 = X[x0_mask]
    X1 = X[x1_mask]


    knn0.fit(X0)
    knn1.fit(X1)

    _, idx0 = knn0.kneighbors(X1)
    _, idx1 = knn1.kneighbors(X0)

    # could be generalized to k-nearest neighbors matching
    idx0 = idx0[:, 0]
    idx1 = idx1[:, 0]


    y0 = data["Y"][x0_mask].iloc[idx0]
    y1 = data["Y"][x1_mask].iloc[idx1]

    # pandas series subtraction is not element-wise, so we need to do this

    y0 = y0.values - data["Y"][x1_mask].values
    y1 = data["Y"][x0_mask].values - y1.values


    return (y0.sum() + y1.sum()) / (len(y0) + len(y1)) 




def calculate_ATE(data, method="IPW", **kwargs):
    if method == "IPW":
        return calculate_IPW_ATE(data, **kwargs)
    elif method == "S-learner":
        return calculate_S_learner_ATE(data, **kwargs)
    elif method == "T-learner":
        return calculate_T_learner_ATE(data, **kwargs)
    elif method == "Matching":
        return calculate_matching_ATE(data, **kwargs)
    else:
        raise ValueError("method must be 'IPW', 'S-learner', 'T-learner', or 'Matching'")

