import pandas as pd
import os
import ast


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'license'), ('artist', 'bio'),
                   ('album', 'type'), ('album', 'information')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def path_from_id(id, prefix='/Users/colinfahy/personal/tf-sandbox/cnn/data/fma_small'):
    id = str(id).zfill(6)
    return os.path.join(prefix, id[0:3], id + '.mp3')


def get_dataset():
    filepath = '/Users/colinfahy/personal/tf-sandbox/cnn/data/fma_metadata/tracks.csv'
    tracks = load(filepath)

    small = tracks['set', 'subset'] <= 'small'
    train = tracks['set', 'split'] == 'training'
    # val = tracks['set', 'split'] == 'validation'
    # test = tracks['set', 'split'] == 'test'
    genres_filepath = '/Users/colinfahy/personal/tf-sandbox/cnn/data/fma_metadata/genres.csv'
    genres = load(genres_filepath)
    labels = {row[0]: {'index': i, 'name': row[3], 'id': row[0]} for i, row in enumerate(genres.itertuples())}

    y_train = tracks.loc[small & train, ('track', 'genres_all')].values
    x_train_ids = tracks.loc[small & train].index.values
    x_train_files = [path_from_id(x) for x in x_train_ids]
    return x_train_files, y_train, labels
