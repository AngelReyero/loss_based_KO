import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import toeplitz
from sklearn.linear_model import (LassoCV, LinearRegression, LassoLarsCV, Lasso,
                                  LogisticRegression, LogisticRegressionCV,
                                  ElasticNetCV, LassoLars)
from sklearn.covariance import (GraphicalLassoCV, empirical_covariance,
                                ledoit_wolf)
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.utils import check_random_state
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
import random



def _bhq_threshold(pvals, fdr=0.1):
    """
    From HiDimStats
    Standard Benjamini-Hochberg
    for controlling False discovery rate

    Calculate threshold for standard Benjamini-Hochberg procedure
    :footcite:`benjamini1995controlling,bhy_2001` for False Discovery Rate (FDR)
    control.

    Parameters
    ----------
    pvals : 1D ndarray
        Array of p-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

    References
    ----------
    .. footbibliography::
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        threshold = pvals_sorted[selected_index]
    else:
        threshold = -1.0
    return threshold


def stat_coefficient_diff(X, X_tilde, y, estimator, fdr, preconfigure_estimator=None):
    """
    from HiDimStats
    Compute the Lasso Coefficient-Difference (LCD) statistic by comparing original and knockoff coefficients.

    This function fits a model on the concatenated original and knockoff features, then
    calculates test statistics based on the difference between coefficient magnitudes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original feature matrix.

    X_tilde : ndarray of shape (n_samples, n_features)
        Knockoff feature matrix.

    y : ndarray of shape (n_samples,)
        Target values.

    estimator : estimator object
        Scikit-learn estimator with fit() method and coef_ attribute.
        Common choices include LassoCV, LogisticRegressionCV.

    fdr : float
        Target false discovery rate level between 0 and 1.

    preconfigure_estimator : callable, default=None
        Optional function to configure estimator parameters before fitting.
        Called with arguments (estimator, X, X_tilde, y).

    Returns
    -------
    test_score : ndarray of shape (n_features,)
        Feature importance scores computed as |beta_j| - |beta_j'|
        where beta_j and beta_j' are original and knockoff coefficients.

    ko_thr : float
        Knockoff threshold value used for feature selection.

    selected : ndarray
        Indices of features with test_score >= ko_thr.

    Notes
    -----
    The test statistic follows Equation 1.7 in Barber & Candès (2015) and
    Equation 3.6 in Candès et al. (2018).
    """
    n_samples, n_features = X.shape
    X_ko = np.column_stack([X, X_tilde])
    if preconfigure_estimator is not None:
        estimator = preconfigure_estimator(estimator, X, X_tilde, y)
    estimator.fit(X_ko, y)
    if hasattr(estimator, "coef_"):
        coef = np.ravel(estimator.coef_)
    elif hasattr(estimator, "best_estimator_") and hasattr(
        estimator.best_estimator_, "coef_"
    ):
        coef = np.ravel(estimator.best_estimator_.coef_)  # for CV object
    else:
        raise TypeError("estimator should be linear")
    # Equation 1.7 in barber2015controlling or 3.6 of candes2018panning
    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    # Compute the threshold level and selecte the important variables
    ko_thr = knockoff_threshold(test_score, fdr=fdr)
    selected = np.where(test_score >= ko_thr)[0]

    return test_score, ko_thr, selected


def knockoff_threshold(test_score, fdr=0.1):
    """
    From HiDimStats
    Calculate the knockoff threshold based on the procedure stated in the article.

    Original code:
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistic.

    fdr : float
        Desired controlled FDR (false discovery rate) level.

    Returns
    -------
    threshold : float or np.inf
        Threshold level.
    """
    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    threshold_mesh = np.sort(np.abs(test_score[test_score != 0]))
    np.concatenate(
        [[0], threshold_mesh, [np.inf]]
    )  # if there is no solution, the threshold is inf
    # find the right value of t for getting a good fdr
    # Equation 1.8 of barber2015controlling and 3.10 in Candès 2018
    threshold = 0.0
    for threshold in threshold_mesh:
        false_pos = np.sum(test_score <= -threshold)
        selected = np.sum(test_score >= threshold)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            break
    return threshold


def simu_data(n, p, rho=0.25, snr=2.0, sparsity=0.06, effect=1.0, seed=None):
    """Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix.
    Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Number of non-null
    k = int(sparsity * p)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(p)
    Sigma = toeplitz(rho ** np.arange(0, p))  # covariance matrix of X
    # X = np.dot(np.random.normal(size=(n, p)), cholesky(Sigma))
    X = rng.multivariate_normal(mu, Sigma, size=(n))
    # Generate the response from a linear model
    blob_indexes = np.linspace(0, p - 6, int(k/5), dtype=int)
    non_zero = np.array([np.arange(i, i+5) for i in blob_indexes])
    # non_zero = rng.choice(p, k, replace=False)
    beta_true = np.zeros(p)
    beta_true[non_zero] = effect
    eps = rng.standard_normal(size=n)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, Sigma


# covariance matrice 
def ind(i,j,k):
    # separates &,n into k blocks
    return int(i//k==j//k)
# One Toeplitz matrix  
def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def GenToysDataset(n=1000, d=10, cor='toep', y_method="nonlin", k=2, mu=None, rho_toep=0.6, sparsity=0.1, seed=0, snr=2):
    """
    Generate a synthetic toy dataset for regression tasks.

    Parameters:
    -----------
    n : int, optional (default=1000)
        Number of samples.
    d : int, optional (default=10)
        Number of features.
    cor : str, optional (default='toep')
        Type of correlation among features. Options:
        - 'iso': Isotropic normal distribution.
        - 'cor': Correlated features using matrix U.
        - 'toep': Toeplitz covariance structure.
    y_method : str, optional (default='nonlin')
        Method for generating target variable y. Options:
        - 'williamson': Quadratic function of first two features.
        - 'hidimstats': High-dimensional sparse regression.
        - 'nonlin': Nonlinear interaction of first five features.
        - 'nonlin2': Extended nonlinear interactions with additional terms.
        - 'lin': Linear combination of first two features.
        - 'poly': Polynomial interactions of randomly selected features.
    k : int, optional (default=2)
        Parameter for correlation matrix U when cor='cor'.
    mu : array-like or None, optional (default=None)
        Mean vector for multivariate normal distribution.
    rho_toep : float, optional (default=0.6)
        Correlation coefficient for Toeplitz covariance matrix.
    sparsity : float, optional (default=0.1)
        Proportion of nonzero coefficients in high-dimensional regression.
    seed : int, optional (default=0)
        Random seed for reproducibility.
    snr : float, optional (default=2)
        Signal-to-noise ratio for high-dimensional regression.

    Returns:
    --------
    X : ndarray of shape (n, d)
        Feature matrix.
    y : ndarray of shape (n,)
        Target variable.
    true_imp : ndarray of shape (d,)
        Binary array indicating which features are truly important.
    """
    np.random.seed(seed)
    true_imp = np.zeros(d)
    
    if y_method == "williamson":
        X1, X2 = np.random.uniform(-1, 1, (2, n))
        X = np.column_stack((X1, X2))
        y = (25/9) * X1**2 + np.random.normal(0, 1, n)
        return X, y, np.array([1, 0])
    
    if y_method == "hidimstats":
        X, y, _, non_zero_index, _ = simu_data(n, d, rho=rho_toep, sparsity=sparsity, seed=seed, snr=snr)
        true_imp[non_zero_index] = 1
        return X, y, true_imp
    
    mu = np.zeros(d) if mu is None else mu
    X = np.zeros((n, d))
    
    if cor == 'iso':
        X = np.random.normal(size=(n, d))
    elif cor == 'cor':
        U = np.array([[ind(i, j, k) for j in range(d)] for i in range(d)]) / np.sqrt(k)
        X = np.random.normal(size=(n, d)) @ U + mu
    elif cor == 'toep':
        X = np.random.multivariate_normal(mu, toep(d, rho_toep), size=n)
    else:
        raise ValueError("Invalid correlation type. Choose from 'iso', 'cor', or 'toep'.")
    
    if y_method == "nonlin":
        y = X[:, 0] * X[:, 1] * (X[:, 2] > 0) + 2 * X[:, 3] * X[:, 4] * (X[:, 2] <= 0)
        true_imp[:5] = 1
    elif y_method == "nonlin2":
        y = (X[:, 0] * X[:, 1] * (X[:, 2] > 0) + 2 * X[:, 3] * X[:, 4] * (X[:, 2] <= 0)
             + X[:, 5] * X[:, 6] / 2 - X[:, 7]**2 + X[:, 9] * (X[:, 8] > 0))
        true_imp[:10] = 1
    elif y_method == "lin":
        y = 2 * X[:, 0] + X[:, 1]
        true_imp[:2] = 1
    elif y_method == "poly":
        rng = np.random.RandomState(seed)
        non_zero_index = rng.choice(d, int(sparsity * d), replace=False)
        poly_transformer = PolynomialFeatures(degree=3, interaction_only=True)
        features = poly_transformer.fit_transform(X[:, non_zero_index])
        coef_features = np.random.choice([-1, 1], features.shape[1])
        y = np.dot(features, coef_features)
        true_imp[non_zero_index] = 1
    else:
        raise ValueError("Invalid y_method. Choose from 'williamson', 'hidimstats', 'nonlin', 'nonlin2', 'lin', or 'poly'.")
    
    return X, y, true_imp




def GenSynthDataset(
    n=300,
    d=50,
    setting="adjacent",
    sparsity=0.25,
    cor="toep",
    rho=0.6,
    seed=0
):
    """
    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of predictors.
    setting : str
        One of:
            - 'adjacent'
            - 'spaced'
            - 'sinusoidal'
            - 'hidim'
            - 'nongauss'
            - 'poly'

    sparsity : float
        Proportion of nonzero signal features (used in sparse settings).

    cor : str
        Correlation structure for X:
            - 'iso'   : independent features
            - 'toep'  : Toeplitz correlation with parameter rho
            - 'ar1'   : AR(1) correlation
            - 'block' : blockwise correlation

    rho : float
        Correlation parameter for Toeplitz, AR(1), or block structures.

    seed : int
        Random seed.

    Returns
    -------
    X : array, shape (n, d)
        Feature matrix.
    y : array, shape (n,)
        Response vector.
    true_imp : array, shape (d,)
        Indicator of truly relevant predictors.
    """
    rng = np.random.default_rng(seed)
    true_imp = np.zeros(d)

    # ------------------------------------------------------------
    # 1. Build covariance matrix
    # ------------------------------------------------------------
    def make_cov(d, cor, rho):
        if cor == "iso":
            return np.eye(d)

        elif cor == "toep":
            # Toeplitz: rho^{|i-j|}
            idx = np.arange(d)
            return rho ** np.abs(idx[:, None] - idx[None, :])

        elif cor == "ar1":
            idx = np.arange(d)
            return rho ** np.abs(idx[:, None] - idx[None, :])

        elif cor == "block":
            b = max(2, d // 5)
            M = np.eye(d)
            for start in range(0, d, b):
                end = min(start+b, d)
                M[start:end, start:end] = rho
                np.fill_diagonal(M[start:end, start:end], 1.0)
            return M

        else:
            raise ValueError("Invalid cor setting.")

    cov = make_cov(d, cor, rho)

    # ------------------------------------------------------------
    # 2. Generate X
    # ------------------------------------------------------------
    if setting == "nongauss":
        # heavy-tailed → correlating through Cholesky
        Xraw = rng.standard_t(df=3, size=(n, d))
        L = np.linalg.cholesky(cov)
        X = Xraw @ L.T
    else:
        X = rng.multivariate_normal(np.zeros(d), cov, size=n)

    # ------------------------------------------------------------
    # 3. Generate y depending on setting
    # ------------------------------------------------------------

    # ---------- Adjacent Support --------------------------------
    if setting == "adjacent":
        k = max(1, int(sparsity * d))
        true_imp[:k] = 1
        beta = np.linspace(1, 2, k)
        y = X[:, :k] @ beta + rng.normal(scale=1, size=n)

    # ---------- Spaced Support ----------------------------------
    elif setting == "spaced":
        stride = max(1, int(1 / sparsity))
        idx = np.arange(0, d, stride)[:max(1, int(sparsity * d))]
        true_imp[idx] = 1
        beta = rng.uniform(1, 2, size=len(idx))
        y = X[:, idx] @ beta + rng.normal(scale=1, size=n)

    # ---------- Sinusoidal Setting -------------------------------
    elif setting == "sinusoidal":
        # overrides X distribution
        X = rng.uniform(0, 2*np.pi, size=(n, d))
        m = min(d, 10)
        true_imp[:m] = 1
        y = np.sum(np.sin(X[:, :m]), axis=1) + rng.normal(scale=0.2, size=n)
# ---------- Sinusoidal Setting -------------------------------
    elif setting == "sin":
        # overrides X distribution
        X = rng.uniform(-np.pi, np.pi, size=(n, d))
        m = min(d, 10)
        true_imp[:m] = 1
        y = np.sum(np.sin(X[:, :m]), axis=1) + rng.normal(scale=0.2, size=n)
    # ---------- High-Dimensional Sparse --------------------------
    elif setting == "hidim":
        k = max(1, int(sparsity * d))
        idx = rng.choice(d, k, replace=False)
        true_imp[idx] = 1
        beta = rng.normal(scale=1, size=k)
        y = X[:, idx] @ beta + rng.normal(scale=1, size=n)

    # ---------- Non-Gaussian Heavy-Tailed X ----------------------
    elif setting == "nongauss":
        k = max(1, int(sparsity * d))
        idx = np.arange(k)
        true_imp[idx] = 1
        beta = rng.normal(scale=1, size=k)
        y = X[:, idx] @ beta + rng.normal(scale=1, size=n)

    # ---------- Polynomial Interactions --------------------------
    elif setting == "poly":
        k = max(1, int(sparsity * d))
        idx = rng.choice(d, k, replace=False)
        true_imp[idx] = 1

        poly = PolynomialFeatures(degree=3, include_bias=False)
        XP = poly.fit_transform(X[:, idx])
        coef = rng.choice([-1, 1], size=XP.shape[1])
        y = XP @ coef + rng.normal(scale=1, size=n)

    else:
        raise ValueError("Unknown setting.")

    return X, y, true_imp


def _estimate_distribution(X, shrink=True, cov_estimator='ledoit_wolf', n_jobs=1):
    """
    Adapted from hidimstat: https://github.com/ja-che/hidimstat
    """
    alphas = [1e-3, 1e-2, 1e-1, 1]

    mu = X.mean(axis=0)
    Sigma = empirical_covariance(X)

    if shrink or not _is_posdef(Sigma):

        if cov_estimator == 'ledoit_wolf':
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]

        elif cov_estimator == 'graph_lasso':
            model = GraphicalLassoCV(alphas=alphas, n_jobs=n_jobs)
            Sigma_shrink = model.fit(X).covariance_

        else:
            raise ValueError('{} is not a valid covariance estimated method'
                             .format(cov_estimator))

        return mu, Sigma_shrink

    return mu, Sigma


def _is_posdef(X, tol=1e-14):
    """Check a matrix is positive definite by calculating eigenvalue of the
    matrix. Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples x n_features)
        Matrix to check

    tol : float, optional
        minimum threshold for eigenvalue

    Returns
    -------
    True or False
    """
    eig_value = np.linalg.eigvalsh(X)
    return np.all(eig_value > tol)


def _get_single_clf_ko(X, j, method="lasso"):
    """
    Fit a single classifier to predict the j-th variable from all others.

    Args:
        X : input data
        j (int): variable index
        method (str, optional): Classifier used. Defaults to "lasso".

    Returns:
        pred: Predicted values for variable j from all others.
    """
    
    n, p = X.shape
    idc = np.array([i for i in np.arange(0, p) if i != j])

    if method == "lasso":
        lambda_max = np.max(np.abs(np.dot(X[:, idc].T, X[:, j]))) / (2 * (p - 1))
        alpha = (lambda_max / 100)
        clf = Lasso(alpha)
    
    if method == "logreg_cv":
        clf = LogisticRegressionCV(cv=5, max_iter=int(10e4), n_jobs=-1)

    if method == "xgb":
        clf = xgb.XGBRegressor(n_jobs=-1)

    clf.fit(X[:, idc], X[:, j])
    pred = clf.predict(X[:, idc])
    return pred


def _get_samples_ko(X, pred, j, discrete=False, adjust_marg=False, seed=None):
    """

    Generate a Knockoff for variable j.

    Args:
        X : input data
        pred (array): Predicted Xj using all other variables
        j (int): variable index
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
        seed (int, optional): seed. Defaults to None.

    Returns:
        sample: Knockoff for variable j.
    """
    np.random.seed(seed)
    n, p = X.shape

    residuals = X[:, j] - pred
    indices_ = np.arange(residuals.shape[0])
    np.random.shuffle(indices_)

    sample = pred + residuals[indices_]

    if adjust_marg:
        sample = _adjust_marginal(sample, X[:, j], discrete=discrete)

    return sample[np.newaxis].T


def _adjust_marginal(v, ref, discrete=False):
    """
    Make v follow the marginal of ref.
    """
    if discrete:
        sorter = np.argsort(v)
        sorter_ = np.argsort(sorter)
        return np.sort(ref)[sorter_]
    
    else:
        G = ECDF(ref)
        F = ECDF(v)

        unif_ = F(v)

        # G_inv = np.argsort(G(ref))
        G_inv = monotone_fn_inverter(G, ref)

        return G_inv(unif_)
    

def conditional_sequential_gen_ko(X, preds, n_jobs=1, discrete=False, adjust_marg=True, seed=None):
    """
    Generate Knockoffs for all variables in X.

    Args:
        X : input data
        preds (array): Predicted values for all variables
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
        seed (int, optional): seed. Defaults to None.
    Returns:
        samples: Knockoffs for all variables in X.
    """
    
    rng = check_random_state(seed)
    n, p = X.shape

    samples = np.hstack(Parallel(n_jobs=n_jobs)(delayed(
        _get_samples_ko)(X, preds[j], j, discrete=discrete, adjust_marg=adjust_marg) for j in tqdm(range(p))))
    
    return samples





def bootstrap_var(imp_list, n_groups=30, size_group=50):
    """
    Compute the variance of bootstrapped importance estimations.

    Parameters:
    -----------
    imp_list : list or array-like
        List of importance values.
    n_groups : int, optional (default=30)
        Number of bootstrap samples to generate.
    size_group : int, optional (default=50)
        Size of each bootstrap sample.

    Returns:
    --------
    float
        Variance of the estimated importance.
    """
    estim_imp = [np.mean(random.choices(imp_list, k=size_group)) for _ in range(n_groups)]
    return np.var(estim_imp)

