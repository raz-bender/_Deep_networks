import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X, self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        N = X.shape[0]
        # w = (X^T*X + λ * I)^-1 * X^T * y
        lambda_I = self.reg_lambda * np.identity(X.shape[1])  # λ * I (regularization term)
        lambda_I[0][0] = 0
        w_opt = np.linalg.inv(X.T @ X + N*lambda_I) @ X.T @ y

        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    if feature_names is None:
        feature_names = [feature for feature in df.columns if feature != target_name]

    #for each feature I want to take the prediction of it with respect to traget
    X = df[feature_names].values        #get feature values in a new Matrix X
    Y = df[target_name].values          #get target values as a vector Y
    y_pred = model.fit_predict(X , Y)   #calc l_w for mse + |w|
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        N = X.shape[0]
        bias_col = np.ones((N, 1))
        xb = np.hstack((bias_col, X))
        # ========================

        return xb

#remove
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
# import pandas as pd
class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        self.poly_feature_adder = PolynomialFeatures(degree=self.degree, include_bias=False)
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        X_transformed = np.delete(X, 4, axis=1)  # deletes CHAS

        # apply log to 'CRIM' (1) and 'LSTAT' (-1)
        X_transformed[:, 1] = np.log(np.abs(X_transformed[:, 1]) + 0.001) #CRIM +0.001 for 0 values
        X_transformed[:, -1] = np.log(np.abs(X_transformed[:, -1]) + 0.001) #LSTAT


        # printing
        # column_names = [f"F{i}" for i in range(X_transformed.shape[1])]
        # # Convert ndarray to DataFrame for plotting
        # df = pd.DataFrame(X_transformed, columns=column_names)
        #
        # # Plot scatter matrix
        # scatter_matrix(df, figsize=(20, 20), diagonal='hist', alpha=0.9, marker='o', grid=True, s=5)
        #
        # plt.suptitle("Scatter Matrix of Features", fontsize=2, y=0.95)  # Add a title
        # plt.show()
        #delete

        X_transformed = self.poly_feature_adder.fit_transform(X_transformed)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    #calculate phi for every feature with traget_feature
    correlations = {}
    for col in df.columns:
        if col != target_feature:
            correlations[col] = df[col].corr(df[target_feature])
    #sort by dis(phi , 1)
    sorted_phis = sorted(correlations.items(), key=lambda phi: 1-abs(phi[1]))

    #return only the top n features
    top_n_features = [feature[0] for feature in sorted_phis[:n]]
    top_n_corr = [feature[1] for feature in sorted_phis[:n]]

    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    N = y.shape[0]
    score = np.sum((y - y_pred)**2)
    # ========================
    return score/(1*N) ###check about the def od mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    y_bar = np.mean(y)
    residual = np.sum((y - y_pred) ** 2)
    #                  /
    sum_of_squares = np.sum((y - y_bar) ** 2)

    r2 = 1 - (residual / sum_of_squares)
    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    kf = sklearn.model_selection.KFold(n_splits=k_folds)
    opt_mse = float("inf")

    for degree in degree_range:
        for reg_lambda in lambda_range:  # foreach combination of the hyperparams
            current_mse = []

            for train_index, validation_index in kf.split(X):  # splits to train and validation sets
                X_train, X_validation = X[train_index], X[validation_index]
                y_train, y_validation = y[train_index], y[validation_index]

                # Set the hyperparameters lambda and degree
                model.set_params(
                    bostonfeaturestransformer__degree=degree,
                    linearregressor__reg_lambda=reg_lambda,
                )
                # Fit-predict with the validation set
                y_pred = model.fit(X_train, y_train).predict(X_validation)

                # take MSE for the curr fold
                mse = mse_score(y_validation, y_pred)
                current_mse.append(mse)
                # end loop folds

            # Average Mse for the fold
            avg_mse = np.mean(current_mse)

            # take the hyperParams that did best
            if avg_mse <= opt_mse:
                opt_mse = avg_mse
                best_params = {
                    "bostonfeaturestransformer__degree": degree,
                    "linearregressor__reg_lambda": reg_lambda,
                }

    # ========================

    return best_params
