import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import savgol_filter

plt.style.use('seaborn-paper')

def grid_dimensions(N):
    cols = np.ceil(np.sqrt(N))
    rows = np.ceil(N / cols)
    return int(rows), int(cols)


def plot_computation_times(times_dict, title='Title', xlabel='x', ylabel='y', loc='upper left'):
    fig, ax = plt.subplots()

    for name, times in times_dict.items():
        x = list(range(1, len(times) + 1))
        ax.plot(x, times, label=name)

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc=loc)
    # plt.savefig(title, dpi=500)
    plt.show()


def plot_class_distribution(labels_dict, title='Title', xlabel='x', ylabel='y', loc='upper right'):
    rows, cols = grid_dimensions(len(labels_dict))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    ax = None
    for i, (name, labels) in enumerate(labels_dict.items()):
        if ax:
            ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(gs[i])
        
        bincounts = np.bincount(labels)
        ax.plot(np.unique(labels), bincounts, label=name)

        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=[0, None])
        ax.legend(loc=loc)

    fig.suptitle(title)
    fig.tight_layout()
    # plt.savefig(title, dpi=500)
    plt.show()


def plot_membership_distribution(labels_dict, title='Title', xlabel='x', ylabel='y', loc='upper right'):
    rows, cols = grid_dimensions(len(labels_dict))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    ax = None
    for i, (name, labels) in enumerate(labels_dict.items()):
        if ax:
            ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(gs[i])
        
        bincounts = np.bincount(labels)
        bincounts = np.bincount(bincounts)[:40]
        ax.bar(list(range(len(bincounts))), height=bincounts, width=0.8, label=name)
        
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(loc=loc)

    fig.suptitle(title)
    fig.tight_layout()
    # plt.savefig(title, dpi=500)
    plt.show()


def read_csvs(names, filenames):
    labels = {}
    times = {}
    for name, filename in zip(names, filenames):
        columns = ['filename', 'race', 'gender', 'age', 'label', 'matchtime']
        df = pd.read_csv(
            filename, header=None, index_col=False, names=columns)
        labels[name] = df['label']
        times[name] = savgol_filter(df['matchtime'], window_length=201, polyorder=3)
        print(f'Read {len(df)} records from {filename}')

    return labels, times


if __name__ == '__main__':
    names = ['HC 0.5', 'HC2 0.85-0.65', 'HCNorm', 'HCNewF']
    directory = './streamface/stats_cluster/'
    filenames = [
        directory + 'sky_hc5.csv',
        directory + 'sky_hc2.csv',
        directory + 'sky_hc3.csv',
        directory + 'sky_hc4.csv',
    ]

    labels, times = read_csvs(names, filenames)

    plot_computation_times(times, 'Computation Time Plot', 'Computation Number', 'Time per Computation (s)')

    plot_class_distribution(labels, 'Class Distribution', 'Class Labels', 'Number of Faces in Class')

    plot_membership_distribution(labels, 'Class Membership Distribution', 'Number of Faces in Class', 'Number of Classes')

