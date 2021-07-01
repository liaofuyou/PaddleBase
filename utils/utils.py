from paddlenlp.datasets import MapDataset


def load_local_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab
