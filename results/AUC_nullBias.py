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
    'Semi_KO': 'orange',
    'CPI_KO_Wilcox': 'orange',
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

    'Semi_KO': 'o',
    'CPI_KO_Wilcox': 'D',

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

    'Semi_KO': (2,4),
    'CPI_KO_Wilcox': (2,1,2),

}




# LINEAR FIGURES
p =100
cor=0.6
alpha = 0.05

parallel = True


csv_files = glob.glob(f"csv/CPI_KO_hidimstats_p{p}_cor{cor}*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# Display the first few rows of the DataFrame
print(df.head())

auc_scores = []
null_imp = []
non_null = []
power = []
type_I = []

def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])
mat = toep(p, cor)
sobol_index = np.zeros(p)
for i in range(p):
    Sigma_without_j = np.delete(mat, i, axis=1)
    Sigma_without_jj = np.delete(Sigma_without_j, i, axis=0)
    sobol_index[i] = (
        mat[i, i]
        - Sigma_without_j[i, :]
        @ np.linalg.inv(Sigma_without_jj)
        @ Sigma_without_j[i, :].T
    )
        

for index, row in df.iterrows():
    y_pred = row.filter(like="imp_V").values
    pval = row.filter(like="pval").values
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))
    non_null.append(np.mean(abs(y_pred[y==1]-sobol_index[y==1])))
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp
df['non_null'] = non_null
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
df['method'] = df['method'].replace('CPI_KO_ST', 'Semi_KO')





sns.set_style("white")

fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})


# AUC
methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W', 'Semi_KO'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='AUC', hue='method', palette=palette, ax=ax[0])  # Left subplot

ax[0].set_xscale('log')
ax[0].tick_params(axis='x', labelsize=15)  
ax[0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[0].set_ylabel(f'AUC', fontsize=20)
ax[0].legend().remove()

# Bias null covariates
sns.lineplot(data=filtered_df, x='n', y='null_imp', hue='method', palette=palette, ax=ax[1])  # Right subplot


# Format bottom-right subplot
ax[1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1].tick_params(axis='y', labelsize=15) 
ax[1].set_xlabel(r'')
ax[1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[1].legend().remove()



# Time
sns.lineplot(data=filtered_df, x='n', y='tr_time', hue='method', palette=palette, ax=ax[2])  


ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].tick_params(axis='x', labelsize=15)  
ax[2].tick_params(axis='y', labelsize=15) 
ax[2].set_xlabel(r'')
ax[2].set_ylabel(f'Time(s)', fontsize=20)
ax[2].legend().remove()
ax[2].legend(
    title="Method",
    fontsize=15,
    title_fontsize=15,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5)
)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"new_figures/AUC_hidim_p{p}_cor{cor}.pdf", bbox_inches="tight")


# POLYNOMIAL FIGURES

p =50
cor=0.6
alpha=0.05
csv_files = glob.glob(f"csv/CPI_KO_poly_p{p}_cor{cor}*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)




# Display the first few rows of the DataFrame
print(df.head())

auc_scores = []
null_imp = []
non_null = []
power = []
type_I = []


for index, row in df.iterrows():
    y_pred = row.filter(like="imp_V").values
    pval = row.filter(like="pval").values
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



df['AUC'] = auc_scores
df['null_imp'] = null_imp
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
df['method'] = df['method'].replace('CPI_KO_ST', 'Semi_KO')


sns.set_style("white")

fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})


df = df[df['method'].isin(markers.keys())]

methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W', 'Semi_KO'] 
filtered_df = df[df['method'].isin(methods_to_plot)]

# AUC
sns.lineplot(data=filtered_df, x='n', y='AUC', hue='method', palette=palette, ax=ax[0])  # Left subplot

ax[0].set_xscale('log')
ax[0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[0].set_ylabel(f'AUC', fontsize=20)
ax[0].legend().remove()


# Bias null covariates
sns.lineplot(data=filtered_df, x='n', y='null_imp', hue='method', palette=palette, ax=ax[1])  # Right subplot


ax[1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  
ax[1].tick_params(axis='y', labelsize=15) 
ax[1].set_xlabel(r'')
ax[1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[1].legend().remove()


plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"figures/CPI_KO_inference_poly_p{p}_cor{cor}_auc.pdf", bbox_inches="tight")


# Time
sns.lineplot(data=filtered_df, x='n', y='tr_time', hue='method', palette=palette, ax=ax[2])  # Left subplot


# Format bottom-left subplot
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[2].tick_params(axis='y', labelsize=15) 
ax[2].set_xlabel(r'')
ax[2].set_ylabel(f'Time(s)', fontsize=20)# left subplot)
ax[2].legend().remove()
ax[2].legend(
    title="Method",
    fontsize=15,
    title_fontsize=15,
    loc="center left",
    bbox_to_anchor=(1.05, 0.5)
)
# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)

plt.savefig(f"new_figures/AUC_poly_p{p}_cor{cor}.pdf", bbox_inches="tight")
