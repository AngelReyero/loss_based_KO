from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import pandas as pd
from hidimstat.statistical_tools.multiple_testing import fdp_power

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


palette = {
    'CPI_KO_Wilcox': 'orange',
    'dCRT': 'magenta',
    'HRT': 'gold',
    "Knockoff": 'blue',
    'CPI_KO': 'orange',
}
markers = {
    'CPI_KO_Wilcox': '*',
    'dCRT': '*',
    'HRT': '*',
    "Knockoff": "o",
    'CPI_KO': "o",    

}


dashes = {
    'CPI_KO_Wilcox': (3, 1, 3),
    'dCRT': (3, 1, 3),
    'HRT': (3, 1, 3),
    "Knockoff": (1, 1),
    'CPI_KO': (1, 1),
}



cor=0.6

y_method = 'poly'
fdr=0.1
ns = [100, 200, 500]
p=int(max(ns)/2)
csv_files = glob.glob(f"csv/KO/{y_method}_cor{cor}*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# Display the first few rows of the DataFrame
print(df.head())

powers = []
fdps = []
auc_scores = []

methods_BH = {'CPI_KO_Wilcox', "dCRT", "HRT"}

for index, row in df.iterrows():
    n = int(row["n"])
    p=int(n/2)
    imp = row.filter(like="imp").values[:p]
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

    fdp, power = fdp_power(selected, non_zero_index)
    fdps.append(fdp)
    powers.append(power)
    




# Add the AUC scores as a new column to the DataFrame
df['power'] = powers
df['fdp'] = fdps
df['AUC'] = auc_scores

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.set_style("white")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
ax = ax.flatten()


# AUC

sns.lineplot(data=df, x='n', y='AUC', hue='method', markers=markers, dashes=dashes,palette=palette,style='method', ax=ax[0])  # Left subplot

ax[0].set_xscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'AUC', fontsize=20)
ax[0].legend().remove()


# Time
sns.lineplot(data=df, x='n', y='tr_time', hue='method', palette=palette,markers=markers, dashes=dashes, style='method', ax=ax[1])  # left subplot


ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].tick_params(axis='x', labelsize=15)  
ax[1].tick_params(axis='y', labelsize=15) 
ax[1].set_xlabel(r'')
ax[1].set_ylabel(f'Time(s)', fontsize=20)
ax[1].legend().remove()

# Power
sns.lineplot(data=df, x='n', y='power', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[2])  # Center subplot


ax[2].set_xscale('log')
ax[2].tick_params(axis='x', labelsize=15)  
ax[2].tick_params(axis='y', labelsize=15) 
ax[2].set_xlabel(r'')
ax[2].set_ylabel(f'Power', fontsize=20)
ax[2].legend().remove()

# Type-I error
sns.lineplot(data=df, x='n', y='fdp', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[3])  # Right subplot


# Format bottom-right subplot
ax[3].set_xscale('log')
ax[3].tick_params(axis='x', labelsize=15)  
ax[3].tick_params(axis='y', labelsize=15) 
ax[3].set_xlabel(r'')
ax[3].set_ylabel(f'FDP', fontsize=20)
ax[3].legend(
    title="Method",
    fontsize=15,
    title_fontsize=15,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5)
)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"figures/KO_{y_method}_p{p}_cor{cor}.pdf", bbox_inches="tight")

