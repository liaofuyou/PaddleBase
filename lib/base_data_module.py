import abc
from functools import partial

from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import PretrainedTokenizer


class BaseDataModule:

    def __init__(self, tokenizer: PretrainedTokenizer, batch_size, max_seq_length, is_predict=False):
        self.is_predict = is_predict
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.predict_ds = None

        if is_predict:
            return

        # 第一步： 加载数据集（训练集、验证集、测试集）
        self.train_ds, self.dev_ds, self.test_ds = self.load_dataset()

        # 第二步： tokenize
        self._tokenize()

        # 第三步： 构造 dataloader
        self.train_dataloader, self.dev_dataloader, self.test_data_loader = self._get_dataloader()

    # 第一步： 加载数据集（训练集、验证集、测试集）]
    @abc.abstractmethod
    def load_dataset(self):
        pass

    # 第二步： tokenize
    def _tokenize(self):
        # 转换函数（ 文本->Token 编号 ）
        trans_fn = partial(self.convert_example)

        # 转换
        self.train_ds.map(trans_fn)
        self.dev_ds.map(trans_fn)
        self.test_ds.map(trans_fn)

    # 第三步： 构造 dataloader
    def _get_dataloader(self):
        # BatchSampler
        train_batch_sampler = DistributedBatchSampler(self.train_ds, batch_size=self.batch_size, shuffle=True)
        dev_batch_sampler = BatchSampler(self.dev_ds, batch_size=self.batch_size, shuffle=False)
        test_batch_sampler = BatchSampler(self.test_ds, batch_size=self.batch_size, shuffle=False)

        # Train DataLoader
        train_dataloader = DataLoader(dataset=self.train_ds,
                                      batch_sampler=train_batch_sampler,
                                      collate_fn=self.batchify_fn(),
                                      return_list=True)
        # Dev DataLoader
        dev_dataloader = DataLoader(dataset=self.dev_ds,
                                    batch_sampler=dev_batch_sampler,
                                    collate_fn=self.batchify_fn(),
                                    return_list=True)
        # Test DataLoader
        test_dataloader = DataLoader(dataset=self.test_ds,
                                     batch_sampler=test_batch_sampler,
                                     collate_fn=self.batchify_fn(),
                                     return_list=True)

        return train_dataloader, dev_dataloader, test_dataloader

    def predict_dataloader(self, data_list: list):
        """predict dataloader"""

        # 数据集
        self.predict_ds = MapDataset(data_list)

        # 转换函数（ 文本->Token 编号 ）
        self.predict_ds.map(self.convert_example)

        # Test DataLoader
        return DataLoader(dataset=self.predict_ds,
                          batch_sampler=BatchSampler(self.predict_ds, shuffle=False),
                          collate_fn=self.batchify_fn(),
                          return_list=True)

    def num_classes(self):
        return 2

    @abc.abstractmethod
    def convert_example(self, example):
        """文本 -> Token Id"""
        pass

    @abc.abstractmethod
    def batchify_fn(self):
        pass
