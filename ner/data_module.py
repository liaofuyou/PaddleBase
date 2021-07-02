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

        fn_list = [
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # seq_len
        ]

        if not is_predict:
            fn_list.append(Pad(axis=0, pad_val=-1))  # labels

        batchify_fn = lambda samples, fn=Tuple(fn_list): [data for data in fn(samples)]

        return batchify_fn

    def num_classes(self):
        return len(self.label_vocab)
