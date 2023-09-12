import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import matplotlib.lines as mlines


def box_whisker_plot(file_list: List):
    "Form a box and whisker plot based on files"

    simulation = []
    ami = []
    total_ami = []
    runtime = []
    for file_name in file_list:
        file = open(file_name, "r")

        for line in file.readlines()[:-1]:
            line_split = line.rstrip().split(" ")
            if len(line_split) != 1:
                simulation.append(line_split[0])
                ami.append(float(line_split[2]))
                total_ami.append(float(line_split[4]))
                runtime.append(float(line_split[6]))

    d = {
        "simulation": simulation,
        "long-term ami": ami,
        "total ami": total_ami,
        "runtime": runtime,
    }
    df = pd.DataFrame(data=d)
    return df


df_800_500 = box_whisker_plot(
    [
        "text_files/temp_log_800_3000_80_5000.txt",
        "text_files/temp_log_800_3000_60_5000.txt",
        "text_files/temp_log_800_3000_100_5000.txt",
    ]
)
df_3000_500 = box_whisker_plot(
    [
        "text_files/temp_log_3000_6000_80_5000.txt",
        "text_files/temp_log_3000_6000_60_5000.txt",
        "text_files/temp_log_3000_6000_100_5000.txt",
    ]
)
df_6000_500 = box_whisker_plot(
    [
        "text_files/temp_log_6000_10000_80_5000.txt",
        "text_files/temp_log_6000_10000_60_5000.txt",
        "text_files/temp_log_6000_10000_100_5000.txt",
    ]
)
df_10000_500 = box_whisker_plot(
    [
        "text_files/temp_log_10000_15000_80_5000.txt",
        "text_files/temp_log_10000_15000_60_5000.txt",
        "text_files/temp_log_10000_15000_100_5000.txt",
    ]
)
df_15000_500 = box_whisker_plot(
    [
        "text_files/temp_log_15000_20000_60_5000.txt",
        "text_files/temp_log_15000_20000_80_5000.txt",
        "text_files/temp_log_15000_20000_100_5000.txt",
    ]
)
df_20000_500 = box_whisker_plot(
    [
        "text_files/temp_log_20000_25000_80_5000.txt",
        "text_files/temp_log_20000_25000_60_5000.txt",
        "text_files/temp_log_20000_25000_100_5000.txt",
    ]
)
df_25000_500 = box_whisker_plot(
    [
        "text_files/temp_log_25000_35000_80_5000.txt",
        "text_files/temp_log_25000_35000_60_5000.txt",
        "text_files/temp_log_25000_35000_100_5000.txt",
    ]
)

df_list = [
    df_800_500,
    df_3000_500,
    df_6000_500,
    df_10000_500,
    df_15000_500,
    df_20000_500,
    df_25000_500,
]


def return_benchmark_df(method: str) -> Tuple[pd.DataFrame]:
    # prefix = 'benchmark/spatio-temporal-clustering-benchmark/benchmark_ami_scores/'
    prefix = "text_files/"

    if method in ["dbscan_baseline"]:
        suffix_800 = "_removenoise_static_log_800_3000_"
        suffix_3000 = "_removenoise_static_log_3000_6000_"
        suffix_6000 = "_removenoise_static_log_6000_10000_"
        suffix_10000 = "_removenoise_static_log_10000_15000_"
        suffix_15000 = "_removenoise_static_log_15000_20000_"
        suffix_20000 = "_removenoise_static_log_20000_25000_"
        suffix_25000 = "_removenoise_static_log_25000_35000_"


    else:
        suffix_800 = "sc_removenoise_static_log_800_3000_"
        suffix_3000 = "sc_removenoise_static_log_3000_6000_"
        suffix_6000 = "sc_removenoise_static_log_6000_10000_"
        suffix_10000 = "sc_removenoise_static_log_10000_15000_"
        suffix_15000 = "sc_removenoise_static_log_15000_20000_"
        suffix_20000 = "sc_removenoise_static_log_20000_25000_"
        suffix_25000 = "sc_removenoise_static_log_25000_35000_"

    df_800 = box_whisker_plot(
        [
            prefix + method + suffix_800 + "06.txt",
            prefix + method + suffix_800 + "08.txt",
            prefix + method + suffix_800 + "010.txt",
        ]
    )

    df_3000 = box_whisker_plot(
        [
            prefix + method + suffix_3000 + "06.txt",
            prefix + method + suffix_3000 + "08.txt",
            prefix + method + suffix_3000 + "010.txt",
        ]
    )

    df_6000 = box_whisker_plot(
        [
            prefix + method + suffix_6000 + "06.txt",
            prefix + method + suffix_6000 + "08.txt",
            prefix + method + suffix_6000 + "010.txt",
        ]
    )

    df_10000 = box_whisker_plot(
        [
            prefix + method + suffix_10000 + "06.txt",
            prefix + method + suffix_10000 + "08.txt",
            prefix + method + suffix_10000 + "010.txt",
        ]
    )

    df_15000 = box_whisker_plot(
        [
            prefix + method + suffix_15000 + "06.txt",
            prefix + method + suffix_15000 + "08.txt",
            prefix + method + suffix_15000 + "010.txt",
        ]
    )

    df_20000 = box_whisker_plot(
        [
            prefix + method + suffix_20000 + "06.txt",
            prefix + method + suffix_20000 + "08.txt",
            prefix + method + suffix_20000 + "010.txt",
        ]
    )

    df_25000 = box_whisker_plot(
        [
            prefix + method + suffix_25000 + "06.txt",
            prefix + method + suffix_25000 + "08.txt",
            prefix + method + suffix_25000 + "010.txt",
        ]
    )

    return [df_800, df_3000, df_6000, df_10000, df_15000, df_20000, df_25000]


def return_benchmark_dbscan_baseline(method: str = "dbscan_baseline"):
    prefix = "text_files/"
    suffix_800 = "_removenoise_static_log_800_3000_"
    suffix_3000 = "_removenoise_static_log_3000_6000_"
    suffix_6000 = "_removenoise_static_log_6000_10000_"
    suffix_10000 = "_removenoise_static_log_10000_15000_"
    suffix_15000 = "_removenoise_static_log_15000_20000_"
    suffix_20000 = "_removenoise_static_log_20000_25000_"
    suffix_25000 = "_removenoise_static_log_25000_35000_"

    df_800 = box_whisker_plot(
        [
            prefix + method + suffix_800 + "06.txt",
            prefix + method + suffix_800 + "08.txt",
            prefix + method + suffix_800 + "010.txt",
        ]
    )

    df_3000 = box_whisker_plot(
        [
            prefix + method + suffix_3000 + "06.txt",
            prefix + method + suffix_3000 + "08.txt",
            prefix + method + suffix_3000 + "010.txt",
        ]
    )

    df_6000 = box_whisker_plot(
        [
            prefix + method + suffix_6000 + "06.txt",
            prefix + method + suffix_6000 + "08.txt",
            prefix + method + suffix_6000 + "010.txt",
        ]
    )

    df_10000 = box_whisker_plot(
        [
            prefix + method + suffix_10000 + "06.txt",
            prefix + method + suffix_10000 + "08.txt",
            prefix + method + suffix_10000 + "010.txt",
        ]
    )

    df_15000 = box_whisker_plot(
        [
            prefix + method + suffix_15000 + "06.txt",
            prefix + method + suffix_15000 + "08.txt",
            prefix + method + suffix_15000 + "010.txt",
        ]
    )

    df_20000 = box_whisker_plot(
        [
            prefix + method + suffix_20000 + "06.txt",
            prefix + method + suffix_20000 + "08.txt",
            prefix + method + suffix_20000 + "010.txt",
        ]
    )

    return [df_800, df_3000, df_6000, df_10000, df_15000, df_20000]


def return_vals(method, metric, benchmark_df_function=return_benchmark_df):
    dfs = benchmark_df_function(method)
    df_vals = [df[metric] for df in dfs]
    df_medians = [np.median(df[metric]) for df in dfs]
    df_averages = [np.average(df[metric]) for df in dfs]
    return df_vals, df_medians, df_averages


for metric_proper in ["Long-term AMI", "Total AMI", "Runtime"]:

    metric = metric_proper.lower()

    agg_vals, agg_medians, agg_avgs = return_vals(method="agglomerative", metric=metric)
    dbscan_vals, dbscan_medians, dbscan_avgs = return_vals(
        method="dbscan", metric=metric
    )
    kmeans_vals, kmeans_medians, kmeans_avgs = return_vals(
        method="kmeans", metric=metric
    )
    birch_vals, birch_medians, birch_avgs = return_vals(method="birch", metric=metric)
    # temp = birch_vals[2].to_list()
    # temp.remove(max(birch_vals[2]))
    # birch_vals[2] = temp
    hdbscan_vals, hdbscan_medians, hdbscan_avgs = return_vals(
        method="hdbscan", metric=metric
    )
    dbscan_baseline_vals, dbscan_baseline_medians, dbscan_baseline_avgs = return_vals(
        method="dbscan_baseline",
        metric=metric,
        benchmark_df_function=return_benchmark_dbscan_baseline,
    )

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        1, 6, sharey=True, figsize=(10, 5)
    )
    ax1.boxplot([df[metric] for df in df_list], positions=np.arange(len(df_list)))
    ax1.plot(
        np.arange(len(df_list)),
        [np.median(df[metric]) for df in df_list],
        "-^",
        c="tab:orange",
    )
    ax1.plot(
        np.arange(len(df_list)), [np.average(df[metric]) for df in df_list], "-o", c="b"
    )

    ax2.boxplot(agg_vals, positions=np.arange(len(agg_vals)))
    ax2.plot(np.arange(len(agg_medians)), agg_medians, "-^", c="tab:orange")
    ax2.plot(np.arange(len(agg_avgs)), agg_avgs, "-o", c="b")

    ax3.boxplot(dbscan_vals, positions=np.arange(len(dbscan_vals)))
    ax3.plot(np.arange(len(dbscan_vals)), dbscan_medians, "-^", c="tab:orange")
    ax3.plot(np.arange(len(dbscan_vals)), dbscan_avgs, "-o", c="b")

    ax4.boxplot(kmeans_vals, positions=np.arange(len(kmeans_vals)))
    ax4.plot(np.arange(len(kmeans_medians)), kmeans_medians, "-^", c="tab:orange")
    ax4.plot(np.arange(len(kmeans_medians)), kmeans_avgs, "-o", c="b")

    ax5.boxplot(birch_vals, positions=np.arange(len(birch_vals)))
    ax5.plot(np.arange(len(birch_medians)), birch_medians, "-^", c="tab:orange")
    ax5.plot(np.arange(len(birch_medians)), birch_avgs, "-o", c="b")

    ax6.boxplot(hdbscan_vals, positions=np.arange(len(hdbscan_vals)))
    ax6.plot(np.arange(len(hdbscan_medians)), hdbscan_medians, "-^", c="tab:orange")
    ax6.plot(np.arange(len(hdbscan_medians)), hdbscan_avgs, "-o", c="b")

    ax1.set_xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize =14
    )
    ax2.set_xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize = 14
    )
    ax3.set_xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize = 14
    )
    ax4.set_xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize=14
    )
    ax5.set_xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize=14
    )
    ax6.set_xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 1000, 15000, 20000, 25000],
        rotation=90,
        fontsize=14
    )

    ax1.set_title("STKM")
    ax2.set_title("ST-Agglomerative")
    ax3.set_title("ST-DBSCAN")
    ax4.set_title("ST-KMeans")
    ax5.set_title("ST-BIRCH")
    ax6.set_title("ST-HDBSCAN")

    point_0 = mlines.Line2D(
        [],
        [],
        color="tab:orange",
        marker="^",
        linestyle="-",
        markersize=5,
        label="Median",
    )
    point_1 = mlines.Line2D(
        [], [], color="b", marker="o", linestyle="-", markersize=5, label="Average"
    )
    plt.legend(
        handles=[point_0, point_1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=12,
    )

    fig.suptitle(metric_proper)
    fig.supxlabel("Dataset Size")
    plt.tight_layout()
    plt.yticks(fontsize=14)

    plt.savefig(metric + "_boxplot.pdf", format="pdf")

    plt.figure()
    plt.plot(
        np.arange(len(df_list)),
        [np.median(df[metric]) for df in df_list],
        "-o",
        label="STKM",
    )
    plt.plot(np.arange(len(agg_medians)), agg_medians, "-P", label="ST-Agglomerative")
    plt.plot(np.arange(len(dbscan_vals)), dbscan_medians, "-*", label="ST-DBSCAN")
    plt.plot(np.arange(len(kmeans_medians)), kmeans_medians, "-d", label="ST-KMeans")
    plt.plot(np.arange(len(birch_medians)), birch_medians, "-^", label="ST-BIRCH")
    plt.plot(np.arange(len(hdbscan_medians)), hdbscan_medians, "-s", label="ST-HDBSCAN")

    plt.title("Median " + metric_proper + " vs. Dataset Size", fontsize=16)
    if metric == "long-term ami":
        plt.plot(
            np.arange(len(dbscan_baseline_medians)),
            dbscan_baseline_medians,
            "-v",
            label="Baseline ST-DBSCAN",
        )
        plt.ylim([-0.05, 1.1])
    elif metric == "total ami":
        plt.ylim([-0.05, 0.60])
    plt.xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize=16
    )
    plt.xlabel("Dataset Size", fontsize=16)
    if metric == "runtime":
        plt.ylabel("Median " + metric_proper + " (s)", fontsize=16)
    else:
        plt.ylabel("Median " + metric_proper, fontsize=16)

    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.yticks(fontsize=14)

    plt.savefig(metric + "_median.pdf", format="pdf")

    plt.figure()
    plt.plot(
        np.arange(len(df_list)),
        [np.average(df[metric]) for df in df_list],
        "-o",
        label="STKM",
    )
    plt.plot(np.arange(len(agg_medians)), agg_avgs, "-P", label="ST-Agglomerative")
    plt.plot(np.arange(len(dbscan_vals)), dbscan_avgs, "-*", label="ST-DBSCAN")
    plt.plot(np.arange(len(kmeans_medians)), kmeans_avgs, "-d", label="ST-KMeans")
    plt.plot(np.arange(len(birch_medians)), birch_avgs, "-^", label="ST-BIRCH")
    plt.plot(np.arange(len(hdbscan_medians)), hdbscan_avgs, "-s", label="ST-HDBSCAN")

    plt.title("Average " + metric_proper + " vs. Dataset Size", fontsize=16)
    if metric == "long-term ami":
        plt.plot(
            np.arange(len(dbscan_baseline_medians)),
            dbscan_baseline_avgs,
            "-v",
            label="Baseline ST-DBSCAN",
        )
        plt.ylim([-0.05, 1.1])
    elif metric == "total ami":
        plt.ylim([-0.05, 0.60])
    plt.xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6],
        labels=[800, 3000, 6000, 10000, 15000, 20000, 25000],
        rotation=90,
        fontsize = 16
    )
    plt.xlabel("Dataset Size", fontsize=16)
    if metric == "runtime":
        plt.ylabel("Average " + metric_proper + " (s)", fontsize=16)
    else:
        plt.ylabel("Average " + metric_proper, fontsize=16)

    plt.legend()
    plt.tight_layout()
    plt.yticks(fontsize=14)
    plt.savefig(metric + "_avg.pdf", format="pdf")

# print(upper_quartile = np.percentile(data, 75)

plt.show()
