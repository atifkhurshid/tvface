import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


plt.style.use('seaborn-paper')


def timestamps_to_labels(timestamps, ts_dict):
    labels = [ts_dict[ts] for ts in timestamps]

    return labels


def plot_frames_over_time(
        dfs_dict, ts_dict, xtick_frequency = 15,
        title='Title', xlabel='x', ylabel='y'):

    for i, (name, df) in enumerate(dfs_dict.items()):

        fig, ax = plt.subplots()

        ts = df['Timestamps']
        x = timestamps_to_labels(ts, ts_dict)
        y = df['# Frames']

        ax.bar(x, y)

        skip_xtick = len(x) // xtick_frequency
        ax.set_xticks(x[::skip_xtick])
        ax.set_xticklabels(ts[::skip_xtick], rotation=90)

        ax.set(xlabel=xlabel, ylabel=ylabel)

        namedtitle = title + ' - ' + name
        fig.suptitle(namedtitle)
        fig.tight_layout()
        plt.savefig(namedtitle, dpi=300)
        plt.show()


def plot_cumulative_distribution(
        dfs_dict, ts_dict, xtick_frequency = 15,
        plot_title='Title', xlabel='x', ylabel='y', loc='upper left',
        pie_title='Pie Title'):

    fig, ax = plt.subplots()

    pie_portions = {}

    for i, (name, df) in enumerate(dfs_dict.items()):

        ts = df['Timestamps']
        x = timestamps_to_labels(ts, ts_dict)
        y = np.cumsum(df['# Frames'].to_list())

        pie_portions[name] = y[-1]

        ax.plot(x, y, label=name)

        skip_xtick = len(x) // xtick_frequency
        ax.set_xticks(x[::skip_xtick])
        ax.set_xticklabels(ts[::skip_xtick], rotation=90)

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(loc=loc)

    fig.suptitle(plot_title)
    fig.tight_layout()

    plt.savefig(plot_title, dpi=500)
    plt.show()

    fig, ax = plt.subplots()

    ax.pie(
        pie_portions.values(),
        labels=list(pie_portions.keys()),
        autopct='%1.1f%%',
    )
    ax.axis('equal')

    fig.suptitle(pie_title)
    fig.tight_layout()

    plt.savefig(pie_title, dpi=500)
    plt.show()



def read_csvs(filenames):
    dfs_dict = {}
    timestamps_list = []
    for name, filename in filenames.items():
        df = pd.read_csv(filename, index_col=0)

        df = pd.DataFrame({
            'Timestamps' : df['Timestamps'],
            '# Frames' : df['# Frames'],
        })
        dfs_dict[name] = df
        timestamps_list.extend(df['Timestamps'])

        print(f'Read {len(df)} records from {filename}')

    unique_timestamps = sorted(list(set(timestamps_list)))
    ts_dict = {k : i for i, k in enumerate(unique_timestamps)}

    return dfs_dict, ts_dict


if __name__ == '__main__':
    directory = './streamface/stats_capture/'
    filenames = {
        'ABC News' : directory + 'abcnews_capture_stats.csv',
        'Al-Jazeera' : directory + 'aljazeera_capture_stats.csv',
        'CGTN News' : directory + 'cgtnnews_capture_stats.csv',
        'CNA' : directory + 'cna_capture_stats.csv',
        'DW News' : directory + 'dwnews_capture_stats.csv',
        'France 24' : directory + 'france24_capture_stats.csv',
        'RT News' : directory + 'rtnews_capture_stats.csv',
        'Sky News' : directory + 'skynews_capture_stats.csv',
    }

    dfs_dict, ts_dict = read_csvs(filenames)

    plot_frames_over_time(
        dfs_dict, ts_dict,
        title='Capture Distribution', xlabel='Timestamps', ylabel='Number of Frames',
    )

    plot_cumulative_distribution(
        dfs_dict, ts_dict,
        plot_title='Total Frames Captured', xlabel='Timestamps', ylabel='Number of Frames',
        pie_title='Captured Frames by Channel')
    