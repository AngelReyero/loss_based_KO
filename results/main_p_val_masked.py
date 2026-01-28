from sklearn.metrics import roc_auc_score
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
    'SCPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'Sobol-CPI(1)_n2': 'blue',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'Sobol-CPI(100)': 'purple',
    'SCPI(100)_sqrt': 'purple',
    'Sobol-CPI(100)_n': 'purple',
    'Sobol-CPI(100)_n2': 'purple',
    'Sobol-CPI(10)_bt': 'cyan',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'purple',
    'Sobol-CPI(1)_ST':'brown',
    'SCPI(1)_Wcx':'brown',
    'LOCO_ST':'black',
    'LOCO_Wcx':'black',
    'Semi_KO_ST': 'orange',
    'SKO_Wcx': 'orange',
    'dCRT': 'magenta',
    'HRT': 'cyan',
    'Semi_KO_ST_perm5': 'gold',
    'SKO_Wcx_p5': 'gold',
    'Semi_KO_ST_perm10': 'blue',
    'SKO_Wcx_perm10': 'blue',
}





alpha = 0.05

#settings=["adjacent", 'masked_corr', "interact_oscillatory",  'cond_var', 'label_noise_gate', "spaced", "nongauss", "cos", "sinusoidal", "sin", "interact_sin", "hidim", "poly", "interact_pairwise", "interact_highorder", "interact_latent"]

#settings=['masked_corr', 'single_index_threshold', 'cond_var', 'label_noise_gate',"adjacent", "spaced", "nongauss", "cos", "sinusoidal", "sin", "interact_sin", "hidim", "poly", "interact_pairwise", "interact_highorder", "interact_latent", "interact_oscillatory"]
settings=['masked_corr']

models = ['GB', 'NN', 'RF']
for setting in settings:
    for model in models:
        csv_files = glob.glob(f"res_csv/p_values_perm2_{setting}_{model}_seed*.csv")
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        # Display the first few rows of the DataFrame
        print(f'setting: {setting} and model: {model}')

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
        df['method'] = df['method'].replace('CPI_sqrt', 'SCPI(1)_sqrt')
        df['method'] = df['method'].replace('S-CPI_sqrt', 'Sobol-CPI(10)_sqrt')
        df['method'] = df['method'].replace('S-CPI2_sqrt', 'SCPI(100)_sqrt')
        df['method'] = df['method'].replace('CPI_bt', 'Sobol-CPI(1)_bt')
        df['method'] = df['method'].replace('S-CPI_bt', 'Sobol-CPI(10)_bt')
        df['method'] = df['method'].replace('S-CPI2_bt', 'Sobol-CPI(100)_bt')
        df['method'] = df['method'].replace('LOCO_sqd', 'LOCO_n2')
        df['method'] = df['method'].replace('S-CPI_Wilcox', 'SCPI(1)_Wcx')
        df['method'] = df['method'].replace('S-CPI_ST', 'Sobol-CPI(1)_ST')
        df['method'] = df['method'].replace('CPI_KO_ST', 'Semi_KO_ST')
        df['method'] = df['method'].replace('CPI_KO_Wilcox', 'SKO_Wcx')
        df['method'] = df['method'].replace('CPI_KO_ST_perm10', 'Semi_KO_ST_perm10')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_perm10', 'SKO_Wcx_perm10')
        df['method'] = df['method'].replace('CPI_KO_ST_perm5', 'Semi_KO_ST_perm5')
        df['method'] = df['method'].replace('CPI_KO_Wilcox_perm5', 'SKO_Wcx_p5')

        # Bottom row for the fixed-n setting (boxplots)

        #sns.set(style="whitegrid", font_scale=1.3)
        sns.set_style("white")
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        metrics = ['power', 'type_I', 'AUC']
        labels = ['Power', 'Type-I Error', 'AUC']
        methods_to_keep = ['SCPI(1)_sqrt', 'SCPI(1)_Wcx', 'SCPI(100)_sqrt', 'LOCO-W', 'LOCO_sqrt', 'LOCO_Wcx','dCRT' , 'HRT', 'SKO_Wcx', 'SKO_Wcx_p5']

        df_plot = df[df['method'].isin(methods_to_keep)].copy()
        palette_filtered = {m: palette[m] for m in methods_to_keep if m in palette}

        #method_order = list(df_plot['method'].unique())

        for ax, metric, label in zip(axes, metrics, labels):

            if metric == 'power':
                ax.clear()
                

                power_stats = df_plot.groupby('method')['power'].agg(
                    mean_power='mean',
                    sem_power=lambda x: np.std(x, ddof=1)/np.sqrt(len(x))
                ).reset_index()

                # Horizontal barplot
                sns.barplot(
                    data=power_stats,
                    y='method',
                    x='mean_power',
                    order=methods_to_keep,
                    palette=palette,
                    ax=ax,
                    ci=None
                )

                # Get the positions of each bar (seaborn returns a list of Rectangle objects)
                bars = ax.patches
                for bar, sem in zip(bars, power_stats['sem_power']):
                    # y-center of the bar
                    y = bar.get_y() + bar.get_height() / 2
                    # x-center is bar.get_width() → length of the bar
                    ax.errorbar(
                        x=bar.get_width(),
                        y=y,
                        xerr=[[min(sem, bar.get_width())], [sem]],
                        fmt='none',
                        ecolor='black',
                        capsize=5,
                        lw=1.5
                    )


                
                ax.set_xlabel("Power", fontsize=18)
                ax.set_ylabel("")
            else: 
                sns.boxplot(
                    data=df_plot,
                    y='method',         # horizontal orientation
                    x=metric,
                    order=methods_to_keep,
                    palette=palette,
                    ax=ax,
                    linewidth=1.3,
                    fliersize=2,
                    orient='h'
                )

        # ----------------------------------------------------
        # Y-axis (method names) ONLY on the first subplot
        # ----------------------------------------------------
        for i, (ax, metric, label) in enumerate(zip(axes, metrics, labels)):
            
            if i == 0:
                ax.set_ylabel("", fontsize=1)
                ax.tick_params(axis='y', labelsize=24)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])   # remove method names
                ax.tick_params(axis='y', length=0)  # remove ticks

            if i == 1:
                ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(label, fontsize=24)
            ax.tick_params(axis='x', labelsize=16)
            if i == 2:
                ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)

            if metric == 'tr_time':
                ax.set_xscale('log')

        # ----------------------------------------------------
        # Legend BELOW the full figure (centered)
        # ----------------------------------------------------

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=color, markersize=10, label=method)
            for method, color in palette_filtered.items()
        ]

    

        hrt_r2_mean = df.loc[df['method'] == 'HRT', 'r2_test'].mean()
        semi_ko_r2_mean = df.loc[df['method'] == 'SKO_Wcx', 'r2_test'].mean()  

        # Reserve space at TOP and BOTTOM before adding title + legend
        plt.subplots_adjust(
            top=0.93,      # space for title
            bottom=0.25,   # space for legend
            wspace=0.05
        )

        # ---- SUPTITLE (safe placement) ----
        fig.suptitle(
            f"Inference Results ($R^2$ HRT={hrt_r2_mean:.3f}, $R^2$ SKO={semi_ko_r2_mean:.3f})",
            fontsize=24
        )

        # ---- LEGEND (placed below plots) ----
        #fig.legend(
        #    handles=handles,
        #    loc='lower center',
        #    bbox_to_anchor=(0.5, 0.05),  # inside the space we reserved with bottom=0.20
        #    ncol=4,
        #    title="Methods",
        #    fontsize=15,
        #    title_fontsize=15
        #)

        plt.savefig(f"main_figures/p_values/main_{setting}_{model}.pdf", bbox_inches="tight")
