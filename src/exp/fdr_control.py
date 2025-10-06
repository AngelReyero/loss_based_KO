# COMPARISON OF METHODS CONTROLLING THE FDR


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from cpi_KO import CPI_KO
from sklearn.base import clone
from hidimstat import D0CRT
from pyhrt.hrt import hrt
from hidimstat.knockoffs import model_x_knockoff

from sklearn.model_selection import KFold


import time
import os
from utils import GenToysDataset
from regression_sampler import RegressionSamplerSingle

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convergence rates super_learner")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    return parser.parse_args()


def main(args):
    
    ns = [100, 200, 500]
    for s in args.seeds:
        y_method='poly'
        
        p=int(max(ns)/2)
        sparsity = 0.25
        cor=0.3
        cor_meth='toep'
        snr=2


        n_jobs=10

        rng = np.random.RandomState(s)

        true_importance=np.zeros((len(ns), p))
        estim_imp=np.zeros((5, len(ns), p))
        executation_time = np.zeros((5, len(ns)))
        if y_method=='hidimstats':
            base_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s)
        elif y_method =='poly':
            ntrees = np.arange(100, 300, 100)
            lr = np.arange(.01, .1, .05)
            param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
            base_model = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 3, n_jobs=n_jobs)

        print("Experiment: "+str(s))
        for (i,n) in enumerate(ns):
            print("With N="+str(n))
            p=int(n/2)
            X, y, true_imp = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, rho_toep=cor, sparsity=sparsity, seed=s, snr=snr)
            true_importance[ i, :p]=true_imp

            model = clone(base_model)
            start_time = time.time()
            d0crt = D0CRT(
                estimator=model,
                screening_threshold=None,
                random_state=s,
            )
            d0crt.fit_importance(X, y)
            executation_time[3,  i] = time.time() - start_time
            estim_imp[3,i,:p]= d0crt.pvalues_.reshape((p,))

            model = clone(base_model)
            start_time = time.time()
            model.fit(X, y)
            tr_KO_time = time.time()-start_time

            start_time = time.time()
            cpi_knockoffs= CPI_KO(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s),
                imputation_model_y=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s),
                random_state=s,
                n_jobs=n_jobs)
            cpi_knockoffs.fit(X, y)
            imp_time_CPI_KO = time.time()-start_time


            start_time = time.time()
            cpi_KO_importance = cpi_knockoffs.score(X, y, n_perm=1,  p_val='wilcox')
            executation_time[2,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            estim_imp[2,i, :p]= cpi_KO_importance["pval"].reshape((p,))

            executation_time[1,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            estim_imp[1,i, :p]= cpi_KO_importance["importance"].reshape((p,))

            model = clone(base_model)
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=s)
            model.fit(X_train, y_train)
            tr_time = time.time()-start_time

            start_time = time.time()
            tstat_fn = lambda X_eval: ((y_test - model.predict(X_eval))**2).mean()
            for j in range(p):
                # Run the HRT
                sampler = RegressionSamplerSingle(
                    estimator=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s),
                    X_train=X_train,
                    X_test=X_test,
                    covariate_index=j
                )
                sampler.fit()

                def conditional(X_eval=None):
                    sample = sampler.sampler().ravel()  # shape (n_test,)
                    # HRT expects length(len(lower)+len(upper)) -> here 2
                    probs = np.ones(2)                   # dummy CI
                    return sample, probs
                result = hrt(j,tstat_fn,X_train,X_test=X_test,conditional=conditional)
                estim_imp[4,i, j]= result['p_value']
            executation_time[4,  i] = time.time() - start_time + tr_time


            start_time = time.time()
            selected, test_scores, threshold, X_tildes = model_x_knockoff(
                X,
                y,
                estimator=LassoCV(
                    n_jobs=1,
                    cv=KFold(n_splits=5, shuffle=True, random_state=0),
                    random_state=1,
                ),
                n_bootstraps=1,
                random_state=2,
            )
            executation_time[0,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            estim_imp[0,i, :p]= test_scores.reshape((p,))

    #Save the results
    f_res = pd.DataFrame()

    for i in range(5):
        for j in range(len(ns)):
            f_res1 = {}

            methods = [
                "Knockoff","CPI_KO","CPI_KO_Wilcox",
                'dCRT', 'HRT'
            ]
            f_res1["method"] = [methods[i]]
            f_res1["n"] = ns[j]

            # adapt p dynamically depending on ns[j]
            p_j = int(ns[j]/2)

            for k in range(p_j):
                f_res1[f"tr_V{k}"]  = true_importance[j, k]
                f_res1[f"imp{k}"]  = estim_imp[i, j, k]

            f_res1['tr_time'] = executation_time[i, j]

            f_res1 = pd.DataFrame(f_res1)
            f_res = pd.concat([f_res, f_res1], ignore_index=True)

    csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        f"../../results/csv/KO/{y_method}_cor{cor}_seed{s}.csv"
    ))
    f_res.to_csv(csv_path, index=False)

    print(f_res.head())


if __name__ == "__main__":
    args = parse_args()
    main(args)