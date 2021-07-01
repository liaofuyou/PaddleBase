from functools import partial

from paddle.fluid.dataloader import Dataset
from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddlenlp.data import Stack, Pad, Tuple, np
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import ErnieGramTokenizer


class DataModule:

    def __init__(self, batch_size=32, max_seq_length=128):

        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        # 第一步： 加载数据集（训练集、验证集、测试集）
        self.train_ds, self.dev_ds, self.test_ds = self._load_dataset()

        # 第二步： tokenize
        self._tokenize()

        # 第三步： 构造 dataloader
        self.train_dataloader, self.dev_dataloader, self.test_data_loader = self._get_dataloader()

    # 第一步： 加载数据集（训练集、验证集、测试集）
    @staticmethod
    def _load_dataset():
        return load_dataset("lcqmc", splits=["train", "dev", "test"])

    # 第二步： tokenize
    def _tokenize(self):
        self.tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')

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
                                      collate_fn=self._batchify_fn(),
                                      return_list=True)
        # Dev DataLoader
        dev_dataloader = DataLoader(dataset=self.dev_ds,
                                    batch_sampler=dev_batch_sampler,
                                    collate_fn=self._batchify_fn(),
                                    return_list=True)
        # Test DataLoader
        test_dataloader = DataLoader(dataset=self.test_ds,
                                     batch_sampler=test_batch_sampler,
                                     collate_fn=self._batchify_fn(),
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
                                collate_fn=self._batchify_fn(True),
                                return_list=True)

        return dataloader

    def convert_example(self, example, max_seq_length, is_predict=False):
        """文本 -> Token Id"""

        query, title = example["query"], example["title"]

        encoded_inputs = self.tokenizer(
            text=query, text_pair=title, max_seq_len=max_seq_length)

        # token id
        input_ids = encoded_inputs["input_ids"]
        # segment ids
        token_type_ids = encoded_inputs["token_type_ids"]

        if is_predict:
            return input_ids, token_type_ids
        else:
            label = np.array([example["label"]], dtype="int64")
            return input_ids, token_type_ids, label

    def _batchify_fn(self, is_predict=False):
        if is_predict:
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            ): [data for data in fn(samples)]
        else:
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # labels
            ): [data for data in fn(samples)]

        return batchify_fn

    @staticmethod
    def num_classes():
        return 2
