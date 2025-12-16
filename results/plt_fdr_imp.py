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
    'Knockoff': 'blue',

    'Semi_KO': 'orange',
    'Semi_KO_Wilcox': 'orange',
    'Semi_KO_imp': 'darkorange',
    'Semi_KO_Wilcox_imp': 'saddlebrown',

    'dCRT': 'magenta',

    'HRT': 'gold',
    'HRT_imp': 'darkgoldenrod',
}

markers = {
    'CPI_KO_Wilcox': '*',
    'dCRT': '*',
    'HRT': '*',
    "Knockoff": "o",
    'CPI_KO': "o",    

}


dashes = {
    'Semi_KO_Wilcox': (3, 1, 3),
    'dCRT': (3, 1, 3),
    'HRT': (3, 1, 3),
    "Knockoff": (1, 1),
    'Semi_KO': (1, 1),
}


settings=["cos", "interact_sin", "interact_pairwise", "interact_highorder", "interact_oscillatory", "adjacent", "hidim", "poly", "spaced", "nongauss", "sin", "sinusoidal"]
#settings=["cos", "interact_sin", "interact_pairwise", "interact_highorder", "interact_oscillatory"]
#settings=["adjacent", "spaced", "sinusoidal", "hidim", "nongauss", "poly"]
models=["lasso", "RF", "NN", "GB", "SL"]

fdr=0.2
for setting in settings:
    for model in models:
        csv_files = glob.glob(f"res_csv/KO_imputed_{setting}_{model}_seed*.csv")
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        df['method'] = df['method'].replace('CPI_KO', 'Semi_KO')
        df['method'] = df['method'].replace('CPI_KO_Wilcox', 'Semi_KO_Wilcox')
        df['method'] = df['method'].replace('CPI_KO_imp', 'Semi_KO_imp')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_imp', 'Semi_KO_Wilcox_imp')
        powers = []
        fdps = []
        auc_scores = []

        methods_BH = {'Semi_KO_Wilcox', "dCRT", "HRT"}

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

        sns.set(rc={'figure.figsize': (16, 4)})
        sns.set_style("white")

        # 1 row × 4 columns
        fig, ax = plt.subplots(1, 4, figsize=(20, 4),
                            gridspec_kw={'wspace': 0.2})

        plot_func = sns.boxplot

        # --- 1. Time ---
        plot_func(
            data=df,
            y='method',
            x='tr_time',
            palette=palette,
            orient='h',
            ax=ax[0]
        )
        ax[0].set_ylabel("Method", fontsize=18)
        ax[0].set_xlabel("Time (s)", fontsize=18)
        ax[0].set_xscale("log")
        ax[0].tick_params(axis='y', labelsize=12)
        ax[0].tick_params(axis='x', labelsize=12)

        # --- 2. Power ---
        plot_func(
            data=df,
            y='method',
            x='power',
            palette=palette,
            orient='h',
            ax=ax[1]
        )
        ax[1].set_ylabel("")
        ax[1].set_xlabel("Power", fontsize=18)
        ax[1].set_yticklabels([])
        ax[1].tick_params(axis='x', labelsize=12)

        # --- 3. FDP / Type-I Error ---
        plot_func(
            data=df,
            y='method',
            x='fdp',
            palette=palette,
            orient='h',
            ax=ax[2]
        )
        ax[2].set_ylabel("")
        ax[2].set_xlabel("FDP", fontsize=18)
        ax[2].set_yticklabels([])
        ax[2].tick_params(axis='x', labelsize=12)
        # Add vertical line at FDP = 0.2
        ax[2].axvline(x=0.2, color='red', linestyle='--', linewidth=2)

        # --- 4. AUC ---
        plot_func(
            data=df,
            y='method',
            x='AUC',
            palette=palette,
            orient='h',
            ax=ax[3]
        )
        ax[3].set_ylabel("")
        ax[3].set_xlabel("AUC", fontsize=18)
        ax[3].set_yticklabels([])
        ax[3].tick_params(axis='x', labelsize=12)
        ax[3].axvline(x=0.5, color='red', linestyle='--', linewidth=2)

        # ---- Manual legend at bottom ----
        legend_elements = [
            Line2D([0], [0], marker='s', color=palette[m], label=m,
                markersize=12, linewidth=0) 
            for m in palette.keys()
        ]

        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=len(palette),
            bbox_to_anchor=(0.5, -0.25),  # x=0.5=center, y=-0.05=slightly below figure
            title='Method',
            fontsize=12,
            title_fontsize=14
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # lower bottom margin, more space for legend

        plt.savefig(f"new_figures/KO/KO_imputed_{setting}_{model}.pdf",
                    bbox_inches="tight")
