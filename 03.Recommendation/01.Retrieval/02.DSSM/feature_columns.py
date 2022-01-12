from collections import OrderedDict, namedtuple, defaultdict


class SparseFeat(namedtuple('SparseFeat', ['name', 'feature_size', 'embedding_dim', 'dtype', 'embedding_name'])):
    def __new__(cls, name, feature_size, embedding_dim=4, dtype='int32', embedding_name=None):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(feature_size, 0.25))

        return super(SparseFeat, cls).__new__(cls, name, feature_size, embedding_dim, dtype, embedding_name)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class SeqSparseFeat(namedtuple('SeqSparseFeat', ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    def __new__(cls, sparsefeat, maxlen, combiner='mean', length_name=None):
        return super(SeqSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def feature_size(self):
        return self.sparsefeat.feature_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    def __hash__(self):
        return self.name.__hash__()
