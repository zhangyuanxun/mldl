
from model import DSSM
from data_loader import *


def main():
    movielens_dataset = MovieLensDataset(train_rating_path=RATING_FILE_PATH_TRAIN,
                                         test_rating_path=RATING_FILE_PATH_TEST,
                                         user_path=USER_FILE_PATH,
                                         item_path=ITEM_FILE_PATH)

    train_dataset, test_dataset = movielens_dataset.load_data()
    train_dataset = movielens_dataset.feature_transformation(train_dataset)
    test_dataset = movielens_dataset.feature_transformation(test_dataset)

    model = DSSM(user_feature_columns=movielens_dataset.user_feature_cols,
                 item_feature_columns=movielens_dataset.item_feature_cols)

    for X_train, y_train in iter(train_dataset):
        out = model(X_train)
        print(out)
        break

if __name__ == "__main__":
    main()