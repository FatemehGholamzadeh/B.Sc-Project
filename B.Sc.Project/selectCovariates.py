import numpy as np
import pandas as pd
from pandas import concat
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt


def mutual_information_cal(data_df, t_list):
    mutual_inf_list = []
    for i in range(len(t_list)):
        t1 = t_list[i]
        row_list = [str(0)]
        for k in range(i):
            row_list.append(str(0))
        for j in range(i+1, len(t_list)):
            t2 = t_list[j]
            tmp = data_df[[t1, t2]].copy()
            tmp.dropna(inplace=True)
            tmp1 = tmp[[t1]].values
            tmp2 = np.squeeze(tmp[[t2]].values)
            tmp_mi = mutual_info_regression(tmp1, tmp2)
            row_list.append("{:0.2f}".format(tmp_mi[0]))
        mutual_inf_list.append(row_list)
    return mutual_inf_list


def draw_table(rowLabels, colLabels, content, title, colWidths=None):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_axis_off()
    table = ax.table(
        cellText=content,
        rowLabels=rowLabels,
        colLabels=colLabels,
        rowColours=["palegreen"] * len(rowLabels),
        colColours=["palegreen"] * len(colLabels),
        cellLoc='center',
        loc='upper center',
        colWidths=colWidths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    # table.scale(2, 2)

    ax.set_title(title, fontweight="bold")
    plt.show()


def select_features(df, target, feature_count, strategy):
    # selecting the most correlated features
    # strategy = MC (Most Correlated) or MMI(Most Mutual Information)
    tags = df.columns
    corrMatrix = df.corr().abs()

    cors = []
    mi_list = []
    for t in tags:
        if t != target:
            tmp = df[[t, target]].copy()
            tmp.dropna(inplace=True)
            tmp1 = tmp[[t]].values
            tmp2 = np.squeeze(tmp[[target]].values)
            if target.startswith("PSI") and not t.startswith("PSI"):
                target_endpart = target.split("_")[-1]
                t_endpart = t.split("_")[-1]
                if target_endpart != t_endpart:
                    continue

            if tmp1.shape[0] < 3:
                continue
            tmp_mi = mutual_info_regression(tmp1, tmp2)
            mi_list.append((tmp_mi, t))

        tmp_corr = corrMatrix[target][t]
        if np.isnan(tmp_corr):
            continue
        else:
            cors.append((tmp_corr, t))

    if strategy == "MC":
        corr_list = sorted(cors, key=lambda x: x[0], reverse=True)
        corr_list = corr_list[:feature_count]
        correlated_tags = [t[1] for t in corr_list]
        corrMatrix = corrMatrix.loc[correlated_tags, correlated_tags]
        corr_list = []
        for t1 in correlated_tags:
            tmp_list = []
            for t2 in correlated_tags:
                tmp_corr = corrMatrix[t1][t2]
                tmp_list.append("{:0.2}".format(tmp_corr))
            corr_list.append(tmp_list)
        draw_table(correlated_tags, correlated_tags, corr_list, "correlation matrix for target {}".format(target))
        return correlated_tags
    else:
        mi_list = sorted(mi_list, key=lambda x: x[0], reverse=True)
        mi_list = mi_list[:feature_count]
        mi_tags = [t[1] for t in mi_list]
        mi_tags.append(target)
        mutual_inf_list = mutual_information_cal(df, mi_tags)
        draw_table(mi_tags, mi_tags, mutual_inf_list, "Mutual Information for target {}".format(target))
        print("mi values is: ", mi_tags)
        return mi_tags

# load the dataset
series = pd.read_excel('all_data_with_weather_filled.xlsx')
series= series[:-32]
series = series.drop(["date"], axis = 1)
# series.index = pd.DatetimeIndex(series['date'])
# data_use = series[['PSI_S22','CO_S22','O3_1_S22', 'O3_8_S22', 'O3_S22', 'PM2_5_S22','NO2_S22', 'SO2_S22', 'PM10_S22']]
tags = select_features(series,'O3_8_S5',7,'MC')
print(tags)