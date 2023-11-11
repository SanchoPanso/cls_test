import os
from io import StringIO
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils.cfg import get_opts


def main():
    args = parse_args()
    opts = get_opts(args)
    
    group = args.group
    data_path = os.path.join(opts.datasets_dir, f'{group}.json')

    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.read_json(StringIO(data['data']))
    class_counts = count_classes(df)

    sns.set(style="whitegrid")
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

    # save histogram to png file
    save_dir = os.path.join(opts.data_path, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    fig = ax.get_figure()
    fig.suptitle(group)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{group}_hist.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default='tits_size')

    args = parser.parse_args()
    return args


def count_classes(df):
    # check if first column is 'path'
    if df.columns[0] == 'path':
        df = df.iloc[:, 1:]

    class_counts = df.sum()
    class_counts['trash'] = len(df) - df.any(axis=1).sum()
    return class_counts


if __name__ == "__main__":
    main()
