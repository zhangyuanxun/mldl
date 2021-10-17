import collections

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split

USER_FILE_PATH = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/u.user"
ITEM_FILE_PATH = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/u.item"
RATING_FILE_PATH_TRAIN = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/ua.base"
RATING_FILE_PATH_TEST = "../../../00.Datasets/02.RS/01.MovieLen/ml-100k/ua.test"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class MovieLensDataset:
    def __init__(self, train_rating_path, test_rating_path, user_path, item_path, min_threshold=4):
        self.user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        self.item_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action',
                          'adventure',
                          'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-Noir',
                          'horror',
                          'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
        self.rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        self.numerical_features = ['age', 'timestamp']
        self.categorical_features = ['user_id', 'item_id', 'gender', 'occupation', 'action', 'adventure', 'animation',
                                     'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-Noir',
                                     'horror',
                                     'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
        self.train_rating_path = train_rating_path
        self.test_rating_path = test_rating_path
        self.user_path = user_path
        self.item_path = item_path
        self.num_features = len(self.numerical_features) + len(self.categorical_features)
        self.field_dims = np.zeros(self.num_features, dtype=np.int64)

        item_df = pd.read_csv(ITEM_FILE_PATH, sep='|', names=self.item_cols)
        item_df = item_df.drop(columns=['title', 'release_date', 'video_release_date', 'imdb_url', 'unknown'])

        user_df = pd.read_csv(USER_FILE_PATH, sep='|', names=self.user_cols)
        user_df = user_df.drop(columns=['zip_code'])
        user_df[['age']] = np.round(MinMaxScaler().fit_transform(user_df[['age']]), 2)

        train_df = pd.read_csv(RATING_FILE_PATH_TRAIN, sep='\t', names=self.rating_cols)
        test_df = pd.read_csv(RATING_FILE_PATH_TEST, sep='\t', names=self.rating_cols)

        self.rating_df = pd.concat([train_df, test_df], axis=0)
        self.rating_df[['timestamp']] = np.round(MinMaxScaler().fit_transform(self.rating_df[['timestamp']]), 2)

        # positive sample (rating >=3), negative sample (rating < 3)
        self.rating_df['rating'] = self.rating_df.rating.apply(lambda x: 1 if int(x) >= 3 else 0)

        # merge with user and item
        self.rating_df = self.rating_df.merge(user_df, on='user_id', how='left')
        self.rating_df = self.rating_df.merge(item_df, on='item_id', how='left')

        self.features = self.rating_df.columns.values.tolist()
        self.features.remove('rating')
        feature2idx = {f: i for i, f in enumerate(self.features)}
        feature_cnts = defaultdict(lambda: defaultdict(int))
        for index, row in self.rating_df.iterrows():
            for feat in feature2idx.keys():
                feature_cnts[feature2idx[feat]][row[feat]] += 1

        self.feature_mapper = {i: {feat for feat, c in cnt.items() if
                                   c >= min_threshold or (i == feature2idx['user_id'] or i == feature2idx[
                                       "item_id"])} for i, cnt in feature_cnts.items()}
        self.feature_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in self.feature_mapper.items()}
        self.defaults = {i: len(feat_pos) for i, feat_pos in self.feature_mapper.items()}
        for i, f in self.feature_mapper.items():
            self.field_dims[i] = len(f) + 1          # reserve one for unknown value
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]))

    def load_data(self):
        feature2idx = {f: i for i, f in enumerate(self.features)}
        datasets = None
        labels = None
        for index, row in self.rating_df.iterrows():
            feature = np.array([self.feature_mapper[feature2idx[k]].get(row[k], self.defaults[feature2idx[k]])
                                for k in feature2idx.keys()])
            feature = feature + self.offsets
            feature = feature.reshape(-1, len(feature))

            datasets = np.append(datasets, feature, axis=0) if datasets is not None else feature
            labels = np.append(labels, row['rating']) if labels is not None else np.array(row['rating'])

        X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2,
                                                            random_state=42, shuffle=True)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    movielens_dataset = MovieLensDataset(train_rating_path=RATING_FILE_PATH_TRAIN,
                                         test_rating_path=RATING_FILE_PATH_TEST,
                                         user_path=USER_FILE_PATH,
                                         item_path=ITEM_FILE_PATH)
    movielens_dataset.load_data()
