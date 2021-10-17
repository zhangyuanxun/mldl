import collections


class UserBasedCF:
    def __init__(self, train_data, test_data):
        self.item_users = collections.defaultdict(set)
        self.train_data = train_data
        self.test_data = test_data

    def user_similarity(self):
        pass

