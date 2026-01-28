import seaborn as sns
import matplotlib.pyplot as plt
import glob 
import numpy as np
import pandas as pd


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
    'HRT': 'cyan',
    'Semi_KO_ST_perm5': 'gold',
    'Semi_KO_Wilcox_perm5': 'gold',
    'Semi_KO_ST_perm10': 'blue',
    'Semi_KO_Wilcox_perm10': 'blue',
}





alpha = 0.05

#settings=["adjacent", 'masked_corr', "interact_oscillatory",  'cond_var', 'label_noise_gate', "spaced", "nongauss", "cos", "sinusoidal", "sin", "interact_sin", "hidim", "poly", "interact_pairwise", "interact_highorder", "interact_latent"]

#settings=['masked_corr', 'single_index_threshold', 'cond_var', 'label_noise_gate',"adjacent", "spaced", "nongauss", "cos", "sinusoidal", "sin", "interact_sin", "hidim", "poly", "interact_pairwise", "interact_highorder", "interact_latent", "interact_oscillatory"]
settings=["wdbc", "diabetes", "wine-red", "wine-white"]
models = ['lasso', 'RF','NN', 'GB','SL']
for setting in settings:
    for model in models:
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
        df['method'] = df['method'].replace('CPI_KO_ST_perm10', 'Semi_KO_ST_perm10')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_perm10', 'Semi_KO_Wilcox_perm10')
        df['method'] = df['method'].replace('CPI_KO_ST_perm5', 'Semi_KO_ST_perm5')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_perm5', 'Semi_KO_Wilcox_perm5')

        # Bottom row for the fixed-n setting (boxplots)

        #sns.set(style="whitegrid", font_scale=1.3)
        sns.set_style("white")
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        metrics = ['tr_time', 'power', 'type_I']
        labels = ['Computation Time (s)', 'Power', 'Type-I Error']
        methods_to_keep = ['Sobol-CPI(1)_sqrt', 'Sobol-CPI(1)_Wilcox', 'Sobol-CPI(100)_sqrt', 'LOCO-W', 'LOCO_sqrt', 'LOCO_Wilcox','dCRT' , 'HRT', 'Semi_KO_Wilcox', 'Semi_KO_Wilcox_perm5']

        df_plot = df[df['method'].isin(methods_to_keep)].copy()
        palette_filtered = {m: palette[m] for m in methods_to_keep if m in palette}

        #method_order = list(df_plot['method'].unique())


        for ax, metric, label in zip(axes, metrics, labels):

            df_c = (
                df_plot.groupby(['method', metric])
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
                x=metric,
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
            if metric == 'tr_time':
                ax.set_xscale('log')
                ax.set_ylabel("Methods", fontsize=16)
                ax.tick_params(axis='y', labelsize=12)
            else: 
                ax.set_ylabel("")
                ax.set_yticklabels([])   # remove method names
                ax.tick_params(axis='y', length=0)  # remove ticks

            if metric == 'type_I':
                ax.axvline(0.05, color='red', linestyle='--', linewidth=2)

            ax.set_xlabel(label, fontsize=16)

            
  
        # ----------------------------------------------------
        # Legend BELOW the full figure (centered)
        # ----------------------------------------------------

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=color, markersize=10, label=method)
            for method, color in palette_filtered.items()
        ]

    

        #hrt_r2_mean = df.loc[df['method'] == 'HRT', 'r2_test'].mean()
        #semi_ko_r2_mean = df.loc[df['method'] == 'Semi_KO_Wilcox', 'r2_test'].mean()  

        # Reserve space at TOP and BOTTOM before adding title + legend
        plt.subplots_adjust(
            top=0.93,      # space for title
            bottom=0.25,   # space for legend
            wspace=0.05
        )

        # ---- SUPTITLE (safe placement) ----
        fig.suptitle(
            f"Real Data Results",
            fontsize=18
        )

        # ---- LEGEND (placed below plots) ----
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.05),  # inside the space we reserved with bottom=0.20
            ncol=4,
            title="Methods",
            fontsize=12,
            title_fontsize=13
        )

        plt.savefig(f"main_figures/p_values/real_data/scatter_{setting}_{model}.pdf", bbox_inches="tight")

