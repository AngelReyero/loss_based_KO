from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import pandas as pd


model = 'lasso'
setting = 'adjacent'

palette = {
    'Sobol-CPI(10)': 'cyan',
    'Sobol-CPI(10)_sqrt': 'cyan',
    'Sobol-CPI(10)_n': 'cyan',
    'Sobol-CPI(10)_n2': 'cyan',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'Sobol-CPI(1)_n2': 'blue',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'Sobol-CPI(100)': 'purple',
    'Sobol-CPI(100)_sqrt': 'purple',
    'Sobol-CPI(100)_n': 'purple',
    'Sobol-CPI(100)_n2': 'purple',
    'Sobol-CPI(10)_bt': 'cyan',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'purple',
    'Sobol-CPI(1)_ST':'brown',
    'Sobol-CPI(1)_Wilcox':'brown',
    'LOCO_ST':'black',
    'LOCO_Wilcox':'black',
    'Semi_KO_ST': 'orange',
    'Semi_KO_Wilcox': 'orange',
    'dCRT': 'magenta',
    'HRT': 'gold',
}

markers = {
    'Sobol-CPI(10)':  "o",
    'Sobol-CPI(10)_sqrt': "^",
    'Sobol-CPI(10)_n': "D",
    'Sobol-CPI(10)_bt': '*',
    'Sobol-CPI(10)_n2': 's',
    
    'Sobol-CPI(1)':  "o",
    'Sobol-CPI(1)_sqrt': "^",
    'Sobol-CPI(1)_n': "D",
    'Sobol-CPI(1)_bt': '*',
    'Sobol-CPI(1)_n2': 's',
    
    'Sobol-CPI(100)':  "o",
    'Sobol-CPI(100)_sqrt': "^",
    'Sobol-CPI(100)_n': "D",
    'Sobol-CPI(100)_bt': '*',
    'Sobol-CPI(100)_n2': 's',
    
    'LOCO-W':  "o",
    'LOCO':  "o",
    'LOCO_n': "D",
    'LOCO_sqrt': "^",
    'LOCO_bt': '*',
    'LOCO_n2': 's',

    'Sobol-CPI(1)_ST':'o',
    'Sobol-CPI(1)_Wilcox':'D',
    'LOCO_ST':'o',
    'LOCO_Wilcox':'D',

    'Semi_KO_ST': 'o',
    'Semi_KO_Wilcox': 'D',

    'dCRT': 'o',
    'HRT': 'D',

}


dashes = {
    'Sobol-CPI(10)':  (3, 5, 1, 5),
    'Sobol-CPI(10)_sqrt': (5, 5),
    'Sobol-CPI(10)_n': (1, 1),
    'Sobol-CPI(10)_bt': (3, 1, 3),
    'Sobol-CPI(10)_n2': (2, 4),
    
    'Sobol-CPI(1)':  (3, 5, 1, 5),
    'Sobol-CPI(1)_sqrt': (5, 5),
    'Sobol-CPI(1)_n': (1, 1),
    'Sobol-CPI(1)_bt': (3, 1, 3),
    'Sobol-CPI(1)_n2': (2, 4),
    
    'Sobol-CPI(100)':  (3, 5, 1, 5),
    'Sobol-CPI(100)_sqrt': (5, 5),
    'Sobol-CPI(100)_n': (1, 1),
    'Sobol-CPI(100)_bt': (3, 1, 3),
    'Sobol-CPI(100)_n2': (2, 4),
    
    'LOCO-W':  (3, 5, 1, 5),
    'LOCO':  (3, 5, 1, 5),
    'LOCO_n': (1, 1),
    'LOCO_sqrt': (5, 5),
    'LOCO_bt': (3, 1, 3),
    'LOCO_n2': (2, 4),

    'Sobol-CPI(1)_ST': (2,4),
    'Sobol-CPI(1)_Wilcox':(2,1,2),
    'LOCO_ST':(2,4),
    'LOCO_Wilcox':(2,1,2),

    'Semi_KO_ST': (2,4),
    'Semi_KO_Wilcox': (2,1,2),

    'dCRT': (2,4),
    'HRT': (2,1,2),
}



alpha = 0.05


csv_files = glob.glob(f"res_csv/p_values_{setting}_{model}*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# Display the first few rows of the DataFrame
print(df.head())

auc_scores = []
power = []
type_I = []

        

for index, row in df.iterrows():
    pval = row.filter(like="pval").values
    pval = np.array(pval, dtype=float)
    pval = np.nan_to_num(pval, nan=1.0)
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, 1-pval)
    auc_scores.append(auc)
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['power'] = power
df['type_I'] = type_I

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})


df['method'] = df['method'].replace('CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('S-CPI', 'Sobol-CPI(10)')
df['method'] = df['method'].replace('S-CPI2', 'Sobol-CPI(100)')
df['method'] = df['method'].replace('CPI_n', 'Sobol-CPI(1)_n')
df['method'] = df['method'].replace('S-CPI_n', 'Sobol-CPI(10)_n')
df['method'] = df['method'].replace('S-CPI2_n', 'Sobol-CPI(100)_n')
df['method'] = df['method'].replace('CPI_sqd', 'Sobol-CPI(1)_n2')
df['method'] = df['method'].replace('S-CPI_sqd', 'Sobol-CPI(10)_n2')
df['method'] = df['method'].replace('S-CPI2_sqd', 'Sobol-CPI(100)_n2')
df['method'] = df['method'].replace('CPI_sqrt', 'Sobol-CPI(1)_sqrt')
df['method'] = df['method'].replace('S-CPI_sqrt', 'Sobol-CPI(10)_sqrt')
df['method'] = df['method'].replace('S-CPI2_sqrt', 'Sobol-CPI(100)_sqrt')
df['method'] = df['method'].replace('CPI_bt', 'Sobol-CPI(1)_bt')
df['method'] = df['method'].replace('S-CPI_bt', 'Sobol-CPI(10)_bt')
df['method'] = df['method'].replace('S-CPI2_bt', 'Sobol-CPI(100)_bt')
df['method'] = df['method'].replace('LOCO_sqd', 'LOCO_n2')
df['method'] = df['method'].replace('S-CPI_Wilcox', 'Sobol-CPI(1)_Wilcox')
df['method'] = df['method'].replace('S-CPI_ST', 'Sobol-CPI(1)_ST')
df['method'] = df['method'].replace('CPI_KO_ST', 'Semi_KO_ST')
df['method'] = df['method'].replace('CPI_KO_Wilcox', 'Semi_KO_Wilcox')



# Bottom row for the fixed-n setting (boxplots)

sns.set(style="whitegrid", font_scale=1.3)

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
metrics = ['tr_time', 'power', 'type_I', 'AUC']
labels = ['Computation Time (s)', 'Power', 'Type-I Error', 'AUC']

# We keep the order of methods as they appear in the data (to preserve consistency)
method_order = list(df['method'].unique())

for ax, metric, label in zip(axes, metrics, labels):
    sns.boxplot(
        data=df,
        x='method',
        y=metric,
        order=method_order,
        palette=palette,
        ax=ax,
        linewidth=1.3,
        fliersize=2
    )
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=16)
    ax.tick_params(axis='x', rotation=90)
    if metric == 'tr_time':
        ax.set_yscale('log')

# Single legend (manual, based on your palette)
handles = []
for method, color in palette.items():
    h = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=method)
    handles.append(h)

fig.legend(
    handles=handles,
    loc='upper center',           # use upper/lower center to anchor relative to bbox
    bbox_to_anchor=(0.5, -0.15), # negative y moves it below the figure
    ncol=4,
    title="Methods",
    fontsize=12,
    title_fontsize=13
)

plt.tight_layout()
plt.subplots_adjust(top=0.82, bottom=0.25, wspace=0.3)


hrt_r2_mean = df.loc[df['method'] == 'HRT', 'r2_test'].mean()
semi_ko_r2_mean = df.loc[df['method'] == 'Semi_KO_Wilcox', 'r2_test'].mean()  

fig.suptitle(f"Inference Results (mean HRT={hrt_r2_mean:.3f}, mean Semi_KO={semi_ko_r2_mean:.3f})", fontsize=18)

plt.savefig(f"new_figures/p_values_{setting}_{model}.pdf", bbox_inches="tight")
