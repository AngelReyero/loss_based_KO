from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from utils import _bhq_threshold
from utils import knockoff_threshold



"""
Script to plot the inference figures with AUC, time and (non) null bias in the appendix.  

The first part covers the linear setting, while the second part focuses on the polynomial setting.  
In the linear setting, the bias in the non-null covariates can be computed,  
as the Total Sobol Index is explicitly computable.  
"""


def compute_fdp_power(selected, nonzero):
    """
    selected: array of indices selected by the method
    nonzero: binary array of length p (1 = true signal, 0 = null)
    """

    selected = np.asarray(selected, dtype=int)

    # true positives = selected indices that are true signals
    tp = np.sum(nonzero[selected] == 1) if len(selected) > 0 else 0

    # false positives = selected indices that are nulls
    fp = len(selected) - tp

    # FDP = FP / max(1, total selected)  (avoid divide-by-zero)
    fdp = fp / max(1, len(selected))

    # total number of true signals
    num_signals = np.sum(nonzero)

    # power = TP / (# true signals)   (if no signals, define power = 0)
    if num_signals > 0:
        power = tp / num_signals
    else:
        power = 0.0

    return fdp, power



palette = {
    'SKO_Wcx': 'orange',
    'dCRT': 'magenta',
    'HRT': 'cyan',
    'Knockoff': 'salmon',
    'SKO': 'orange',
    'SKO_p5': 'gold',
    'SKO_Wcx_p5': 'gold',
    'SKO_p10': 'cyan',
    'SKO_Wcx_p10': 'red'
}

markers = {
    'CPI_KO_Wcx': '*',
    'dCRT': '*',
    'HRT': '*',
    "Knockoff": "o",
    'CPI_KO': "o",    

}


dashes = {
    'SKO_Wcx': (3, 1, 3),
    'dCRT': (3, 1, 3),
    'HRT': (3, 1, 3),
    "Knockoff": (1, 1),
    'SKO': (1, 1),
}

settings = ["nongauss"]#["adjacent", "hidim", "poly", "spaced", "nongauss", "sin", "sinusoidal", "cos", "interact_sin", "interact_pairwise", "interact_highorder", "interact_oscillatory"]
models=["RF", "NN", "GB"]

fdr=0.2
for setting in settings:
    for model in models:
        csv_files = glob.glob(f"res_csv/KO_perm2_{setting}_{model}_seed*.csv")
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        df['method'] = df['method'].replace('CPI_KO', 'SKO')
        df['method'] = df['method'].replace('CPI_KO_Wilcox', 'SKO_Wcx')
        df['method'] = df['method'].replace('CPI_KO_perm5', 'SKO_p5')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_perm5', 'SKO_Wcx_p5')
        df['method'] = df['method'].replace('CPI_KO_perm10', 'SKO_p10')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_perm10', 'SKO_Wcx_p10')

        powers = []
        fdps = []
        auc_scores = []

        methods_BH = {'SKO_Wcx', "dCRT", "HRT", 'SKO_Wcx_p5', 'SKO_Wcx_p10'}

        for index, row in df.iterrows():
            n = int(row["n"])
            if setting == 'hidim':
                p = 200
            else:
                p = 50
            imp = row.filter(like="pval").values[:p]
            imp = np.array(imp, dtype=float)  # force numeric
            imp = np.nan_to_num(imp, nan=1.0, posinf=1.0, neginf=1.0)
            non_zero_index = row.filter(like="tr_V").values[:p].astype(int)
            if row["method"] in methods_BH:
                threshold = _bhq_threshold(imp, fdr=fdr)
                selected = np.where(imp <=threshold)[0]
                auc = roc_auc_score(non_zero_index, -imp)
            else:
                ko_thr = knockoff_threshold(imp, fdr=fdr)
                selected = np.where(imp >= ko_thr)[0]
                auc = roc_auc_score(non_zero_index, imp)
            auc_scores.append(auc)
            #print(selected)
            #print(non_zero_index)
            fdp, power = compute_fdp_power(selected, non_zero_index)
            fdps.append(fdp)
            powers.append(power)
            


        # Add the AUC scores as a new column to the DataFrame
        df['power'] = powers
        df['fdp'] = fdps
        df['AUC'] = auc_scores

        # ---- PLOTTING ----

        from matplotlib.lines import Line2D
        methods_to_keep = ['Knockoff', 'dCRT', 'HRT', 'SKO', 'SKO_p5', 'SKO_Wcx', 'SKO_Wcx_p5']
        df_plot = df[df['method'].isin(methods_to_keep)].copy()
        palette_filtered = {m: palette[m] for m in methods_to_keep if m in palette}

        sns.set(rc={'figure.figsize': (12, 4)})
        sns.set_style("white")

        # 1 row × 4 columns
        fig, ax = plt.subplots(1, 3, figsize=(15, 4),
                            gridspec_kw={'wspace': 0.2})

        plot_func = sns.boxplot

        # --- 1. Power ---
        plot_func(
            data=df_plot,
            y='method',
            x='power',
            palette=palette,
            order=methods_to_keep,
            orient='h',
            ax=ax[0]
        )
        ax[0].set_ylabel("")
        ax[0].set_xlabel("Power", fontsize=22)
        ax[0].tick_params(axis='y', labelsize=22)
        ax[0].tick_params(axis='x', labelsize=16)

        # --- 2. FDP / Type-I Error ---
        plot_func(
            data=df_plot,
            y='method',
            x='fdp',
            palette=palette,
            order=methods_to_keep,
            orient='h',
            ax=ax[1]
        )
        ax[1].set_ylabel("")
        ax[1].set_xlabel("FDP", fontsize=22)
        ax[1].set_yticklabels([])
        ax[1].tick_params(axis='x', labelsize=16)
        # Add vertical line at FDP = 0.2
        ax[1].axvline(x=0.2, color='red', linestyle='--', linewidth=2)

        # --- 3. AUC ---
        plot_func(
            data=df_plot,
            y='method',
            x='AUC',
            palette=palette,
            order=methods_to_keep,
            orient='h',
            ax=ax[2]
        )
        ax[2].set_ylabel("")
        ax[2].set_xlabel("AUC", fontsize=22)
        ax[2].set_yticklabels([])
        ax[2].tick_params(axis='x', labelsize=16)
        ax[2].axvline(x=0.5, color='red', linestyle='--', linewidth=2)

        
        hrt_r2_mean = df.loc[df['method'] == 'HRT', 'r2_test'].mean()
        Sko_r2_mean = df.loc[df['method'] == 'SKO', 'r2_test'].mean()  

        plt.subplots_adjust(
            top=0.9,      # space for title
            bottom=0.1,   # space for legend
            wspace=0.05
        )

        # ---- SUPTITLE (safe placement) ----
        fig.suptitle(
            f"Inference Results ($R^2$ HRT={hrt_r2_mean:.3f}, $R^2$ SKO={Sko_r2_mean:.3f})",
            fontsize=22
        )
        plt.savefig(f"main_figures/KO/main_{setting}_{model}.pdf",
                    bbox_inches="tight")
