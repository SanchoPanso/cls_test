import os
from io import StringIO
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default='sex_positions')

    args = parser.parse_args()
    return args


def count_classes(df):
    # check if first column is 'path'
    if df.columns[0] == 'path':
        df = df.iloc[:, 1:]

    class_counts = df.sum()
    class_counts['trash'] = len(df) - df.any(axis=1).sum()
    return class_counts


def main():
    args = parse_args()
    group = args.group

    data_dir = '../DATA/datasets'
    data_path = os.path.join(data_dir, f'{group}.json')

    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.read_json(StringIO(data['data']))

    class_counts = count_classes(df)

    sns.set(style="whitegrid")
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

    # save histogram to png file
    save_dir = '../DATA/figures'
    os.makedirs(save_dir, exist_ok=True)
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_dir, f'{group}_hist.png'))


if __name__ == "__main__":
    main()
