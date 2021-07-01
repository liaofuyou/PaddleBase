from functools import partial

from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddlenlp.data import Stack, Pad, Tuple, np
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepTokenizer


class DataModule:

    def __init__(self, batch_size=32, max_seq_length=128):

        self.batch_size = batch_size

        # 第一步： 加载数据集（训练集、验证集、测试集）
        self.train_ds, self.dev_ds, self.test_ds = self._load_dataset()

        # 第二步： tokenize
        self._tokenize(max_seq_length)

        # 第三步： 构造 dataloader
        self.train_dataloader, self.dev_dataloader, self.test_data_loader = self._get_dataloader()

    # 第一步： 加载数据集（训练集、验证集、测试集）
    @staticmethod
    def _load_dataset():
        return load_dataset("chnsenticorp", splits=["train", "dev", "test"])

    # 第二步： tokenize
    def _tokenize(self, max_seq_length):
        self.tokenizer = SkepTokenizer.from_pretrained("skep_ernie_1.0_large_ch")

        # 转换函数（ 文本->Token 编号 ）
        trans_fn = partial(self.convert_example, max_seq_length=max_seq_length)
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

    def convert_example(self, example, max_seq_length, is_test=False):
        """文本 -> Token Id"""

        encoded_inputs = self.tokenizer(text=example["text"], max_seq_len=max_seq_length)

        # token id
        input_ids = encoded_inputs["input_ids"]
        # segment ids
        token_type_ids = encoded_inputs["token_type_ids"]

        if not is_test:
            # label：情感极性类别
            label = np.array([example["label"]], dtype="int64")
            return input_ids, token_type_ids, label
        else:
            # qid：每条数据的编号
            qid = np.array([example["qid"]], dtype="int64")
            return input_ids, token_type_ids, qid

    def _batchify_fn(self, is_test=False):
        ignore_label = -1
        if is_test:
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # qid
            ): [data for data in fn(samples)]
        else:
            # 训练数据会返回 input_ids, token_type_ids, labels 3 个字段
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # labels
            ): [data for data in fn(samples)]

        return batchify_fn

    def num_classes(self):
        return len(self.train_ds.label_list)
