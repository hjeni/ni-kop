from collections import namedtuple
import os
import pandas as pd

from _utils import measure_time


"""
Suitable for experiment which revolve around a single solver

path: destination where the experiment results are stored
f: function which performs the actual experiment -> pd.DataFrame
kwargs: kwargs of f
"""
Experiment = namedtuple('Experiment', ['path', 'f', 'kwargs'])


def experiment_solo(e: Experiment, allow_loading=True, verbose=False):
    """
    Wraps the single-solver experiment, allows to load result data from a file when the experiment is already conducted

    As an experiment, expects a function which returns a pd.DataFrame
    """
    # try to load data
    if allow_loading and os.path.exists(e.path):
        if verbose:
            print(f'[Experiment] Loading data from CSV.')
        return pd.read_csv(e.path)

    # perform experiment
    if verbose:
        print(f'[Experiment] Starting..')
    df, t = measure_time(e.f)(**e.kwargs)
    if verbose:
        print(f'[Experiment] Time of execution: {t / 1e9:.1f} s')
    # save collected data
    df.to_csv(e.path, index=False)

    return df


"""
Suitable for experiment which use (e.g. compare) multiple solvers

tags: tags of the dataframes
paths: destination where the experiment results are stored, iterable
f: function which performs the actual experiment -> dict({'tag': pd.DataFrame})
kwargs: kwargs of f
"""
MultiExperiment = namedtuple('MultiExperiment', ['tags', 'paths', 'f', 'kwargs'])


def experiment_multi(e: MultiExperiment, allow_loading=True, verbose=False):
    """
    Wraps the multi-solver experiment, allows to load result data from a file when the experiment is already conducted

    As an experiment, expects a function which returns a pd.DataFrame
    """
    dfs_return = {}

    # try to load data
    if allow_loading and all(os.path.exists(p) for p in e.paths):
        if verbose:
            print(f'[Multi-experiment] Loading data from CSV.')
        for tag, path in zip(e.tags, e.paths):
            dfs_return[tag] = pd.read_csv(path)
        return dfs_return

    # perform all experiments
    if verbose:
        print(f'[Multi-experiment] Starting..')
    dfs_dict, t = measure_time(e.f)(**e.kwargs)
    print(f'[Multi-experiment] Time of execution: {t / 1e9:.1f} s')
    if verbose and len(dfs_dict) != len(e.paths):
        print(f'[WARNING: Multi-experiment] more dataframes returned than expected.')
    # iterate & save all data
    for path, e_tag, (tag, df) in zip(e.paths, e.tags, dfs_dict.items()):
        if verbose and e_tag != tag:
            print(f'[WARNING: Multi-experiment] Different tag than expected: {e_tag} | {tag}')
        # save current
        dfs_return[e_tag] = df
        df.to_csv(path, index=False)

    return dfs_return

