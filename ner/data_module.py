from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.transformers import ErnieTokenizer

from lib.base_data_module import BaseDataModule
from utils.utils import load_local_dataset, load_dict


class NerDataModule(BaseDataModule):

    def __init__(self, batch_size=32, max_seq_length=512):

        super().__init__(ErnieTokenizer.from_pretrained('ernie-1.0'), batch_size, max_seq_length)

    def load_dataset(self):
        self.label_vocab = load_dict('./data/waybill/tag.dic')
        return load_local_dataset(
            datafiles=('./data/waybill/train.txt', './data/waybill/dev.txt', './data/waybill/test.txt'))

    def convert_example(self, example, is_predict=False):
        """文本 -> Token Id"""
        tokens, labels = example
        tokenized_input = self.tokenizer(tokens,
                                         return_length=True,
                                         is_split_into_words=True)

        input_ids = tokenized_input['input_ids']
        token_type_ids = tokenized_input['token_type_ids']
        seq_len = tokenized_input['seq_len']

        # Token '[CLS]' and '[SEP]' will get label 'O'
        labels = ['O'] + labels + ['O']

        # 讲label名字换为label Id
        labels = [self.label_vocab[x] for x in labels]

        return input_ids, token_type_ids, seq_len, labels

    def batchify_fn(self, is_predict=False):
        ignore_label = -1
        if is_predict:
            # predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # seq_len
                Pad(axis=0, pad_val=-1)  # labels
            ): fn(samples)
        else:
            # 训练数据会返回 input_ids, token_type_ids, labels 3 个字段
            batchify_fn = lambda samples, fn=Tuple(
                Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
                Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
                Stack(dtype="int64"),  # seq_len
                Pad(axis=0)  # labels
            ): [data for data in fn(samples)]

        return batchify_fn

    def num_classes(self):
        return len(self.label_vocab)
