from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import pandas as pd


"""
Script to plot the inference figures with AUC, time and (non) null bias in the appendix.  

The first part covers the linear setting, while the second part focuses on the polynomial setting.  
In the linear setting, the bias in the non-null covariates can be computed,  
as the Total Sobol Index is explicitly computable.  
"""


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
    'CPI_KO_ST': 'orange',
    'CPI_KO_Wilcox': 'orange',
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

    'CPI_KO_ST': 'o',
    'CPI_KO_Wilcox': 'D',

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

    'CPI_KO_ST': (2,4),
    'CPI_KO_Wilcox': (2,1,2),

    'dCRT': (2,4),
    'HRT': (2,1,2),

}




p =50
cor=0.6
alpha = 0.05

y_method = 'poly'


csv_files = glob.glob(f"csv/n_p_fixed_{y_method}_Var_cor{cor}*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# Display the first few rows of the DataFrame
print(df.head())

power = []
type_I = []


for index, row in df.iterrows():
    pval = row.filter(like="pval").values
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



# Add the AUC scores as a new column to the DataFrame
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



sns.set_style("white")


# Bottom row for the linear setting

fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})



# Time
sns.lineplot(data=df, x='n', y='tr_time', hue='method', palette=palette,markers=markers, dashes=dashes, style='method', ax=ax[0])  # left subplot


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'Time(s)', fontsize=20)
ax[0].legend().remove()

# Power
sns.lineplot(data=df, x='n', y='power', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[1])  # Center subplot


ax[ 1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[1].set_ylabel(f'Power', fontsize=20)
ax[ 1].legend().remove()

# Type-I error
sns.lineplot(data=df, x='n', y='type_I', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[2])  # Right subplot


# Format bottom-right subplot
ax[2].set_xscale('log')
ax[ 2].tick_params(axis='x', labelsize=15)  
ax[ 2].tick_params(axis='y', labelsize=15) 
ax[ 2].set_xlabel(r'')
ax[ 2].set_ylabel(f'Type-I error', fontsize=20)
ax[2].legend(
    title="Method",
    fontsize=15,
    title_fontsize=15,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5)
)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"figures/n_p_fixed_{y_method}_p{p}_cor{cor}.pdf", bbox_inches="tight")


# FILTERED 

mean_typeI = df.groupby(["method", "n"])["type_I"].mean().reset_index()

valid_methods = (
    mean_typeI.groupby("method")["type_I"]
    .apply(lambda x: (x < 0.1).all())
)
valid_methods = valid_methods[valid_methods].index.tolist()

df_filtered = df[df["method"].isin(valid_methods)]

print("Methods kept:", valid_methods)


fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})



# Time
sns.lineplot(data=df_filtered, x='n', y='tr_time', hue='method', palette=palette,markers=markers, dashes=dashes, style='method', ax=ax[0])  # left subplot


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'Time(s)', fontsize=20)
ax[0].legend().remove()

# Power
sns.lineplot(data=df_filtered, x='n', y='power', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[1])  # Center subplot


ax[ 1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[1].set_ylabel(f'Power', fontsize=20)
ax[ 1].legend().remove()

# Type-I error
sns.lineplot(data=df_filtered, x='n', y='type_I', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[2])  # Right subplot


# Format bottom-right subplot
ax[2].set_xscale('log')
ax[ 2].tick_params(axis='x', labelsize=15)  
ax[ 2].tick_params(axis='y', labelsize=15) 
ax[ 2].set_xlabel(r'')
ax[ 2].set_ylabel(f'Type-I error', fontsize=20)
ax[2].legend(
    title="Method",
    fontsize=15,
    title_fontsize=15,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5)
)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"figures/valid_n_p_fixed_{y_method}_p{p}_cor{cor}.pdf", bbox_inches="tight")


