import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import pandas as pd


palette = {
    'SCPI(10)': 'cyan',
    'SCPI(10)_sqrt': 'cyan',
    'SCPI(10)_n': 'cyan',
    'SCPI(10)_n2': 'cyan',
    'SCPI(1)': 'blue',
    'SCPI(1)_sqrt': 'blue',
    'SCPI(1)_n': 'blue',
    'SCPI(1)_n2': 'blue',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'SCPI(100)': 'purple',
    'SCPI(100)_sqrt': 'purple',
    'SCPI(100)_n': 'purple',
    'SCPI(100)_n2': 'purple',
    'SCPI(10)_bt': 'cyan',
    'SCPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'SCPI(100)_bt': 'purple',
    'SCPI(1)_ST':'brown',
    'SCPI(1)_Wcx':'brown',
    'LOCO_ST':'black',
    'LOCO_Wcx':'black',
    'SKO_ST': 'orange',
    'SKO_Wcx': 'orange',
    'dCRT': 'magenta',
    'HRT': 'cyan',
    'SKO_ST_p5': 'gold',
    'SKO_Wcx_p5': 'gold',
    'SKO_ST_p10': 'blue',
    'SKO_Wcx_p10': 'blue',
}





alpha = 0.05

#settings=["adjacent", 'masked_corr', "interact_oscillatory",  'cond_var', 'label_noise_gate', "spaced", "nongauss", "cos", "sinusoidal", "sin", "interact_sin", "hidim", "poly", "interact_pairwise", "interact_highorder", "interact_latent"]

#settings=['masked_corr', 'single_index_threshold', 'cond_var', 'label_noise_gate',"adjacent", "spaced", "nongauss", "cos", "sinusoidal", "sin", "interact_sin", "hidim", "poly", "interact_pairwise", "interact_highorder", "interact_latent", "interact_oscillatory"]
setting = "wdbc"
models = ['RF','NN', 'GB']#,'SL']

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.set_style("white")
fig, axes = plt.subplots(1, 3, figsize=(13.5*0.85, 7.5*0.85))
for ax, model in zip(axes, models):
    csv_files = glob.glob(f"res_csv/real_data/p_values_{setting}_{model}_seed*.csv")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


    # Display the first few rows of the DataFrame
    print(f'setting: {setting} and model: {model}')

            

    imp_cols = [c for c in df.columns if c.startswith("pval")]
    p = len(imp_cols)
    null_idx = p - 1
    #df["null_importance"] = df[f"imp_V{null_idx}"]

    pval_cols = [c for c in df.columns if c.startswith("pval")]
    pval_nonnull = pval_cols[:-1]

    df["power"] = (df[pval_nonnull] < alpha).sum(axis=1)
    df["type_I"] = (df[f"pval{null_idx}"] < alpha).astype(int)



    df['method'] = df['method'].replace('CPI', 'SCPI(1)')
    df['method'] = df['method'].replace('S-CPI', 'SCPI(10)')
    df['method'] = df['method'].replace('S-CPI2', 'SCPI(100)')
    df['method'] = df['method'].replace('CPI_n', 'SCPI(1)_n')
    df['method'] = df['method'].replace('S-CPI_n', 'SCPI(10)_n')
    df['method'] = df['method'].replace('S-CPI2_n', 'SCPI(100)_n')
    df['method'] = df['method'].replace('CPI_sqd', 'SCPI(1)_n2')
    df['method'] = df['method'].replace('S-CPI_sqd', 'SCPI(10)_n2')
    df['method'] = df['method'].replace('S-CPI2_sqd', 'SCPI(100)_n2')
    df['method'] = df['method'].replace('CPI_sqrt', 'SCPI(1)_sqrt')
    df['method'] = df['method'].replace('S-CPI_sqrt', 'SCPI(10)_sqrt')
    df['method'] = df['method'].replace('S-CPI2_sqrt', 'SCPI(100)_sqrt')
    df['method'] = df['method'].replace('CPI_bt', 'SCPI(1)_bt')
    df['method'] = df['method'].replace('S-CPI_bt', 'SCPI(10)_bt')
    df['method'] = df['method'].replace('S-CPI2_bt', 'SCPI(100)_bt')
    df['method'] = df['method'].replace('LOCO_sqd', 'LOCO_n2')
    df['method'] = df['method'].replace('LOCO_Wilcox', 'LOCO_Wcx')
    df['method'] = df['method'].replace('S-CPI_Wilcox', 'SCPI(1)_Wcx')
    df['method'] = df['method'].replace('S-CPI_ST', 'SCPI(1)_ST')
    df['method'] = df['method'].replace('CPI_KO_ST', 'SKO_ST')
    df['method'] = df['method'].replace('CPI_KO_Wilcox', 'SKO_Wcx')
    df['method'] = df['method'].replace('CPI_KO_ST_perm10', 'SKO_ST_p10')
    df['method'] = df['method'].replace('CPI_KO_Wilcox_perm10', 'SKO_Wcx_p10')
    df['method'] = df['method'].replace('CPI_KO_ST_perm5', 'SKO_ST_p5')
    df['method'] = df['method'].replace('CPI_KO_Wilcox_perm5', 'SKO_Wcx_p5')
    df['method'] = df['method'].replace('Semi_KO_ST', 'SKO_ST')
    df['method'] = df['method'].replace('Semi_KO_Wilcox', 'SKO_Wcx')
    df['method'] = df['method'].replace('Semi_KO_ST_perm10', 'SKO_ST_p10')
    df['method'] = df['method'].replace('Semi_KO_Wilcox_perm10', 'SKO_Wcx_p10')
    df['method'] = df['method'].replace('Semi_KO_ST_perm5', 'SKO_ST_p5')
    df['method'] = df['method'].replace('Semi_KO_Wilcox_perm5', 'SKO_Wcx_p5')
    # Bottom row for the fixed-n setting (boxplots)

    #sns.set(style="whitegrid", font_scale=1.3)
    
    #metrics = ['tr_time', 'power', 'type_I']
    #labels = ['Computation Time (s)', 'Power', 'Type-I Error']
    methods_to_keep = ['SCPI(1)_sqrt', 'SCPI(1)_Wcx', 'SCPI(100)_sqrt', 'LOCO-W', 'LOCO_sqrt', 'LOCO_Wcx','dCRT' , 'HRT', 'SKO_Wcx', 'SKO_Wcx_p5']

    df_plot = df[df['method'].isin(methods_to_keep)].copy()
    palette_filtered = {m: palette[m] for m in methods_to_keep if m in palette}

    #method_order = list(df_plot['method'].unique())

    df_c = (
        df_plot.groupby(['method', 'power'])
        .size()
        .reset_index(name='freq')
    )
    gamma = 20           # >1 => bigger differentiation
    df_c['size'] = df_c['freq'] ** gamma
    df_c['method'] = pd.Categorical(df_c['method'],
                        categories=methods_to_keep,
                        ordered=True)

    sns.scatterplot(
        data=df_c,
        x="power",
        y='method',
        size='freq',
        hue_order=methods_to_keep,
        sizes=(1, 500), # min/max point sizes
        hue='method',
        palette=palette_filtered,
        legend=False,
        #order=methods_to_keep,
        ax=ax
    )
    if model == 'RF':
        #ax.set_xscale('log')
        ax.set_ylabel("Methods", fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
    else: 
        ax.set_ylabel("")
        ax.set_yticklabels([])   # remove method names
        ax.tick_params(axis='y', length=0)  # remove ticks


    if model == 'NN':
        ax.set_xlabel("Discoveries", fontsize=16)
    else:
        ax.set_xlabel("")

    df_type1 = (
    df_plot.groupby('method')['type_I']
            .mean()
            #.mul(100)
            .round(3)     # e.g. 4.7%
            .reset_index(name='type_I_pct')
    )
    df_ann = df_type1.copy()
    df_ann['method'] = pd.Categorical(df_ann['method'],
                            categories=methods_to_keep,
                            ordered=True)

    # annotate on the right side of plot
    x_max = df_c['power'].max() if len(df_c)>0 else 0
    if model == 'NN':
        shift = 4.35
    elif model == 'RF':
        shift = 1.75
    elif model == 'GB':
        shift = 2.85
    for _, row in df_ann.iterrows():
        ax.text(
            x=x_max - shift,              # a bit to the right of max power
            y=row['method'],
            s=f"{row['type_I_pct']}",
            va='center',
            fontsize=12,
            color='black'
        )

    # extend xlim so annotation isn't clipped
    ax.set_xlim(-1, x_max )

        

    # ----------------------------------------------------
    # Legend BELOW the full figure (centered)
    # ----------------------------------------------------

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor=color, markersize=10, label=method)
        for method, color in palette_filtered.items()
    ]



    #hrt_r2_mean = df.loc[df['method'] == 'HRT', 'r2_test'].mean()
    #SKO_r2_mean = df.loc[df['method'] == 'SKO_Wcx', 'r2_test'].mean()  

    # Reserve space at TOP and BOTTOM before adding title + legend
    plt.subplots_adjust(
        top=0.93,      # space for title
        bottom=0.25,   # space for legend
        wspace=0.05
    )

    # ---- SUPTITLE (safe placement) ----
    #fig.suptitle(
    #    f"Real Data Results",
    #    fontsize=18
    #)
    model_labels = {
        "RF": "Random Forest",
        "GB": "Gradient Boosting",
        "SL": "Super Learner",
        "NN": "Neural Network"
    }
    ax.set_title(model_labels[model], fontsize=15, pad=15)

    # ---- LEGEND (placed below plots) ----
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),  # inside the space we reserved with bottom=0.20
        ncol=5,
        title="Methods",
        fontsize=12,
        title_fontsize=13
    )

plt.savefig(f"main_figures/p_values/real_data/power_{setting}_{model}.pdf", bbox_inches="tight")




