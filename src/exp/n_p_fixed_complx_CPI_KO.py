import numpy as np
import vimpy
from sobol_CPI import Sobol_CPI
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from loco import LOCO
from cpi_KO import CPI_KO
from sklearn.base import clone
from hidimstat import D0CRT
from pyhrt.hrt import hrt


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
    

    for s in args.seeds:
        p = 50
        y_method='poly'

        ns = [200, 500, 1000, 5000]
        sparsity = 0.25
        cor=0.6
        cor_meth='toep'
        snr=2


        n_cal=10
        n_cal2 = 100
        n_jobs=10

        best_model=None
        dict_model=None

        rng = np.random.RandomState(s)

        true_importance=np.zeros(( len(ns), p))
        p_val=np.zeros((31, len(ns), p))
        executation_time = np.zeros((31, len(ns)))
        # 0 LOCO-W, 1-6 Sobol-CPI(1) (-, sqrt, n, bootstrap, n2), 7-11 Sobol-CPI(10)(-, sqrt, n, bootstrap, n2), 12-16 Sobol-CPI(100) (-, sqrt, n, bootstrap, n2), 17-21 LOCO (-, sqrt, n, bootstrap, n2)
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
            X, y, true_imp = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, rho_toep=cor, sparsity=sparsity, seed=s, snr=snr)
            true_importance[ i]=true_imp

            model = clone(base_model)
            start_time = time.time()
            d0crt = D0CRT(
                estimator=model,
                screening_threshold=None,
                random_state=s,
            )
            d0crt.fit_importance(X, y)
            executation_time[27,  i] = time.time() - start_time
            p_val[27,i]= d0crt.pvalues_.reshape((p,))

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
            cpi_KO_importance = cpi_knockoffs.score(X, y, n_perm=1,  p_val='sign_test')
            executation_time[25,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            p_val[25,i]= cpi_KO_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_KO_importance = cpi_knockoffs.score(X, y, n_perm=1,  p_val='wilcox')
            executation_time[26,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            p_val[26,i]= cpi_KO_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_knockoffs= CPI_KO(
                estimator=model,
                imputation_model=clone(base_model),
                imputation_model_y=clone(base_model),
                random_state=s,
                n_jobs=n_jobs)
            cpi_knockoffs.fit(X, y)
            imp_time_CPI_KO = time.time()-start_time

            start_time = time.time()
            cpi_KO_importance = cpi_knockoffs.score(X, y, n_perm=1,  p_val='sign_test')
            executation_time[29,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            p_val[29,i]= cpi_KO_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_KO_importance = cpi_knockoffs.score(X, y, n_perm=1,  p_val='wilcox')
            executation_time[30,  i] = time.time() - start_time + tr_KO_time + imp_time_CPI_KO
            p_val[30,i]= cpi_KO_importance["pval"].reshape((p,))

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
                p_val[28,i, j]= result['p_value']
            executation_time[28,  i] = time.time() - start_time + tr_time

            start_time = time.time()
            sobol_CPI= Sobol_CPI(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s),
                n_permutations=1,
                random_state=s,
                n_jobs=n_jobs)
            sobol_CPI.fit(X_train, y_train)
            imp_time = time.time()-start_time

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='sign_test')
            executation_time[21,  i] = time.time() - start_time + tr_time + imp_time
            p_val[21,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='wilcox')
            executation_time[22,  i] = time.time() - start_time + tr_time + imp_time
            p_val[22,i]= cpi_importance["pval"].reshape((p,))


            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='emp_var')
            executation_time[1,  i] = time.time() - start_time + tr_time + imp_time
            p_val[1,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_sqrt')
            executation_time[2,  i] = time.time() - start_time + tr_time + imp_time
            p_val[2,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_n')
            executation_time[3, i] = time.time() - start_time + tr_time + imp_time
            p_val[3,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_n', bootstrap=True)
            executation_time[4, i] = time.time() - start_time + tr_time + imp_time
            p_val[4,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=1,  p_val='corrected_sqd', bootstrap=True)
            executation_time[5, i] = time.time() - start_time + tr_time + imp_time
            p_val[5,i]= cpi_importance["pval"].reshape((p,))

            # Sobol-CPI(n_cal)
            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='emp_var')
            executation_time[6,  i] = time.time() - start_time + tr_time + imp_time
            p_val[6,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_sqrt')
            executation_time[7, i] = time.time() - start_time + tr_time + imp_time
            p_val[7,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_n')
            executation_time[8,  i] = time.time() - start_time + tr_time + imp_time
            p_val[8,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_n', bootstrap=True)
            executation_time[9,  i] = time.time() - start_time + tr_time + imp_time
            p_val[9,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_sqd', bootstrap=True)
            executation_time[10, i] = time.time() - start_time + tr_time + imp_time
            p_val[10,i]= cpi_importance["pval"].reshape((p,))

            #Sobol-CPI(ncal2)
            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='emp_var')
            executation_time[11, i] = time.time() - start_time + tr_time + imp_time
            p_val[11,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_sqrt')
            executation_time[12, i] = time.time() - start_time + tr_time + imp_time
            p_val[12,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_n')
            executation_time[13, i] = time.time() - start_time + tr_time + imp_time
            p_val[13,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_n', bootstrap=True)
            executation_time[14, i] = time.time() - start_time + tr_time + imp_time
            p_val[14,i]= cpi_importance["pval"].reshape((p,))

            start_time = time.time()
            cpi_importance = sobol_CPI.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_sqd', bootstrap=True)
            executation_time[15, i] = time.time() - start_time + tr_time + imp_time
            p_val[15,i]= cpi_importance["pval"].reshape((p,))

            #LOCO
            start_time = time.time()
            loco = LOCO(
                estimator=model,
                random_state=s,
                loss=mean_squared_error, 
                n_jobs=n_jobs,
            )
            loco.fit(X_train, y_train)
            tr_loco_time = time.time()-start_time

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='sign_test')
            executation_time[23, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[23,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='wilcox')
            executation_time[24, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[24,i]= loco_importance["pval"].reshape((p,))


            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='emp_var')
            executation_time[16, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[16,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_sqrt')
            executation_time[17, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[17,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_n')
            executation_time[18, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[18,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_n', bootstrap=True)
            executation_time[19, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[19,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            loco_importance = loco.score(X_test, y_test, p_val='corrected_sqd', bootstrap=True)
            executation_time[20, i] = time.time() - start_time + tr_time + tr_loco_time
            p_val[20,i]= loco_importance["pval"].reshape((p,))

            start_time = time.time()
            #LOCO Williamson
            for j in range(p):
                print("covariate: "+str(j))
                if y_method=='hidimstats':
                    model_j=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=s)
                elif y_method =='poly':
                    ntrees = np.arange(100, 300, 100)
                    lr = np.arange(.01, .1, .05)
                    param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
                    model_j = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 3, n_jobs=n_jobs)
                vimp = vimpy.vim(y = y, x = X, s = j, pred_func = model_j, measure_type = "r_squared")
                vimp.get_point_est()
                vimp.get_influence_function()
                vimp.get_se()
                vimp.get_ci()
                vimp.hypothesis_test(alpha = 0.05, delta = 0)
                p_val[0,  i, j]=vimp.p_value_
            executation_time[0, i] = time.time() - start_time 


    #Save the results
    f_res = pd.DataFrame()

    for i in range(31):
        for j in range(len(ns)):
            f_res1 = {}

            methods = [
                "LOCO-W","CPI","CPI_sqrt","CPI_n","CPI_bt","CPI_sqd",
                "S-CPI","S-CPI_sqrt","S-CPI_n","S-CPI_bt","S-CPI_sqd",
                "S-CPI2","S-CPI2_sqrt","S-CPI2_n","S-CPI2_bt","S-CPI2_sqd",
                "LOCO","LOCO_sqrt","LOCO_n","LOCO_bt","LOCO_sqd",
                "S-CPI_ST","S-CPI_Wilcox","LOCO_ST","LOCO_Wilcox",
                "CPI_KO_ST","CPI_KO_Wilcox", 'dCRT', 'HRT',"CPI_KO_CPLX_ST","CPI_KO_CPLX_Wilcox",
            ]
            f_res1["method"] = [methods[i]]
            f_res1["n"] = ns[j]


            for k in range(p):
                f_res1[f"tr_V{k}"]  = true_importance[j, k]
                f_res1[f"pval{k}"]  = p_val[i, j, k]

            f_res1['tr_time'] = executation_time[i, j]

            f_res1 = pd.DataFrame(f_res1)
            f_res = pd.concat([f_res, f_res1], ignore_index=True)

    csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        f"../../results/csv/n_p_fixed_cplx_{y_method}_Var_cor{cor}_seed{s}.csv"
    ))
    f_res.to_csv(csv_path, index=False)

    print(f_res.head())


if __name__ == "__main__":
    args = parse_args()
    main(args)