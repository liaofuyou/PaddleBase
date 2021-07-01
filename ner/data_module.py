from functools import partial

from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.transformers import ErnieTokenizer

from utils.utils import load_local_dataset, load_dict


class DataModule:

    def __init__(self, pretrained_model, batch_size=32):

        self.batch_size = batch_size

        # 第一步： 加载数据集（训练集、验证集、测试集）
        self.train_ds, self.dev_ds, self.test_ds = self._load_dataset()

        # 第二步： tokenize
        self._tokenize(pretrained_model)

        # 第三步： 构造 dataloader
        self.train_dataloader, self.dev_dataloader, self.test_data_loader = self._get_dataloader()

    # 第一步： 加载数据集（训练集、验证集、测试集）
    def _load_dataset(self):
        self.label_vocab = load_dict('../data/waybill/tag.dic')
        return load_local_dataset(
            datafiles=('./data/waybill/train.txt', './data/waybill/dev.txt', './data/waybill/test.txt'))
        # return load_dataset("lcqmc", splits=["train", "dev", "test"])

    # 第二步： tokenize
    def _tokenize(self, pretrained_model):
        self.tokenizer = ErnieTokenizer.from_pretrained(pretrained_model)

        # 转换函数（ 文本->Token 编号 ）
        trans_fn = partial(self.convert_example, max_seq_length=512)
        trans_fn_for_test = partial(trans_fn, is_test=True)

        # 转换
        self.train_ds.map(trans_fn)
        self.dev_ds.map(trans_fn)
        self.test_ds.map(trans_fn_for_test)

    # 第三步： 构造 dataloader
    def _get_dataloader(self):

        # batchify
        train_ds_batchify_fn = self._batchify_fn(False)
        dev_ds_batchify_fn = self._batchify_fn(False)
        test_ds_batchify_fn = self._batchify_fn(True)

        # BatchSampler
        train_batch_sampler = DistributedBatchSampler(self.train_ds, batch_size=self.batch_size, shuffle=True)
        dev_batch_sampler = BatchSampler(self.dev_ds, batch_size=self.batch_size, shuffle=False)
        test_batch_sampler = BatchSampler(self.test_ds, batch_size=self.batch_size, shuffle=False)

        # Train DataLoader
        train_dataloader = DataLoader(dataset=self.train_ds,
                                      batch_sampler=train_batch_sampler,
                                      collate_fn=train_ds_batchify_fn,
                                      return_list=True)
        # Dev DataLoader
        dev_dataloader = DataLoader(dataset=self.dev_ds,
                                    batch_sampler=dev_batch_sampler,
                                    collate_fn=dev_ds_batchify_fn,
                                    return_list=True)
        # Test DataLoader
        test_dataloader = DataLoader(dataset=self.test_ds,
                                     batch_sampler=test_batch_sampler,
                                     collate_fn=test_ds_batchify_fn,
                                     return_list=True)

        return train_dataloader, dev_dataloader, test_dataloader

    def convert_example(self, example, max_seq_length=512, is_test=False):
        """文本 -> Token Id"""
        tokens, labels = example
        tokenized_input = self.tokenizer(tokens,
                                         return_length=True,
                                         is_split_into_words=True)
        # Token '[CLS]' and '[SEP]' will get label 'O'
        labels = ['O'] + labels + ['O']
        tokenized_input['labels'] = [self.label_vocab[x] for x in labels]

        return tokenized_input['input_ids'], tokenized_input['token_type_ids'], tokenized_input['seq_len'], \
               tokenized_input['labels']

    def _batchify_fn(self, is_test=False):
        ignore_label = -1
        if is_test:
            # predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # seq_len
                Pad(axis=0, pad_val=ignore_label)  # labels
            ): [data for data in fn(samples)]
        else:
            # 训练数据会返回 input_ids, token_type_ids, labels 3 个字段
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # seq_len
                Pad(axis=0, pad_val=ignore_label)  # labels
            ): [data for data in fn(samples)]

        return batchify_fn

    def num_classes(self):
        return len(self.label_vocab)
