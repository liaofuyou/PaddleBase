import abc
from functools import partial

from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import PretrainedTokenizer


class BaseDataModule:

    def __init__(self, tokenizer: PretrainedTokenizer, batch_size, max_seq_length):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

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
        trans_fn = partial(self.convert_example, max_seq_length=self.max_seq_length)

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

    # predict dataloader
    def predict_dataloader(self, lines: list):
        dataset = MapDataset(lines)

        # 转换函数（ 文本->Token 编号 ）
        trans_fn = partial(self.convert_example,
                           max_seq_length=self.max_seq_length,
                           is_predict=True)

        dataset.map(trans_fn)

        # Test DataLoader
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=BatchSampler(dataset, shuffle=False),
                                collate_fn=self.batchify_fn(True),
                                return_list=True)

        return dataloader

    def num_classes(self):
        return len(self.train_ds.label_list)

    @abc.abstractmethod
    def convert_example(self, example, max_seq_length, is_predict=False):
        """文本 -> Token Id"""
        pass

    @abc.abstractmethod
    def batchify_fn(self, is_predict=False):
        pass
