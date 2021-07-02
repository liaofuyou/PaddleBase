from paddlenlp.data import Stack, Pad, Tuple, np
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieGramTokenizer

from lib.base_data_module import BaseDataModule


class PointwiseMatchingDataModule(BaseDataModule):

    def __init__(self, batch_size=32, max_seq_length=128):

        super().__init__(ErnieGramTokenizer.from_pretrained('ernie-gram-zh'), batch_size, max_seq_length)

    def load_dataset(self):
        return load_dataset("lcqmc", splits=["train", "dev", "test"])

    def convert_example(self, example, is_predict=False):
        """转换：文本 -> Token Id"""

        query, title = example["query"], example["title"]

        encoded_inputs = self.tokenizer(
            text=query, text_pair=title, max_seq_len=self.max_seq_length)

        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]

        if is_predict:
            return input_ids, token_type_ids
        else:
            label = np.array([example["label"]], dtype="int64")
            return input_ids, token_type_ids, label

    def batchify_fn(self, is_predict=False):
        """对齐"""

        fn_list = [
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id)  # token_type_ids
        ]
        if not is_predict:
            fn_list.append(Stack(dtype="int64"))  # labels

        batchify_fn = lambda samples, fn=Tuple(fn_list): [data for data in fn(samples)]

        return batchify_fn
