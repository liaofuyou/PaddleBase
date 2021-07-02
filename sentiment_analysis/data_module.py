from paddlenlp.data import Stack, Pad, Tuple, np
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepTokenizer

from lib.base_data_module import BaseDataModule


class SentimentAnalysisDataModule(BaseDataModule):

    def __init__(self, batch_size=32, max_seq_length=128):

        super().__init__(SkepTokenizer.from_pretrained("skep_ernie_1.0_large_ch"), batch_size, max_seq_length)

    # 加载数据集（训练集、验证集、测试集）
    def load_dataset(self):
        return load_dataset("chnsenticorp", splits=["train", "dev", "test"])

    def convert_example(self, example, is_predict=False):
        """转换：文本 -> Token Id"""

        encoded_inputs = self.tokenizer(text=example["text"], max_seq_len=self.max_seq_length)

        input_ids = encoded_inputs["input_ids"]  # token id
        token_type_ids = encoded_inputs["token_type_ids"]  # segment ids

        if not is_predict:
            # label：情感极性类别
            label = np.array([example["label"]], dtype="int64")
            return input_ids, token_type_ids, label
        else:
            # qid：每条数据的编号
            qid = np.array([example["qid"]], dtype="int64")
            return input_ids, token_type_ids, qid

    def batchify_fn(self, is_predict=False):
        """对齐"""

        batchify_fn = lambda samples, fn=Tuple([
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64")]  # [labels] when train/dev/test OR [qid] when predict
        ): [data for data in fn(samples)]

        return batchify_fn
