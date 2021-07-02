from paddlenlp.data import Stack, Pad, Tuple, np
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieGramTokenizer, ErnieTokenizer

from lib.base_data_module import BaseDataModule


class ReadingComprehensionDataModule(BaseDataModule):

    def __init__(self, batch_size=32, max_seq_length=512):

        super().__init__(ErnieTokenizer.from_pretrained('ernie-1.0'), batch_size, max_seq_length)

    def load_dataset(self):
        return load_dataset("dureader_robust", splits=["train", "dev", "test"])

    def convert_example(self, examples, is_predict=False):
        """转换：文本 -> Token Id"""
        """
        {
             'id': '7de192d6adf7d60ba73ba25cf590cc1e', 
             'title': '',
             'context': '选择燃气热水器时，一定要关注这几个问题：1、出水稳定性要好，不能出现忽热忽冷的现象2、快速到达设定的需求水温3、操作要智能、方便4、安全性要好，要装有安全报警装置 市场上燃气热水器品牌众多，购买时还需多加对比和仔细鉴别。方太今年主打的磁化恒温热水器在使用体验方面做了全面升级：9秒速热，可快速进入洗浴模式；水温持久稳定，不会出现忽热忽冷的现象，并通过水量伺服技术将出水温度精确控制在±0.5℃，可满足家里宝贝敏感肌肤洗护需求；配备CO和CH4双气体报警装置更安全（市场上一般多为CO单气体报警）。另外，这款热水器还有智能WIFI互联功能，只需下载个手机APP即可用手机远程操作热水器，实现精准调节水温，满足家人多样化的洗浴需求。当然方太的磁化恒温系列主要的是增加磁化功能，可以有效吸附水中的铁锈、铁屑等微小杂质，防止细菌滋生，使沐浴水质更洁净，长期使用磁化水沐浴更利于身体健康。',
             'question': '燃气热水器哪个牌子好', 
             'answers': ['方太'], 
             'answer_starts': [110]
         }
        """
        doc_stride = 128

        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            stride=doc_stride,
            max_seq_len=self.max_seq_length)

        # Let's label those examples!
        for i, tokenized_example in enumerate(tokenized_examples):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_example["input_ids"]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offsets = tokenized_example['offset_mapping']

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_example['token_type_ids']

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_example['overflow_to_sample']
            answers = examples[sample_index]['answers']
            answer_starts = examples[sample_index]['answer_starts']

            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Minus one more to reach actual text
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1

        return tokenized_examples

        # contexts = [examples[i]['context'] for i in range(len(examples))]
        # questions = [examples[i]['question'] for i in range(len(examples))]
        #
        # query, title = examples["query"], examples["title"]
        #
        # encoded_inputs = self.tokenizer(
        #     text=query, text_pair=title, max_seq_len=self.max_seq_length)
        #
        # input_ids = encoded_inputs["input_ids"]
        # token_type_ids = encoded_inputs["token_type_ids"]
        #
        # if is_predict:
        #     return input_ids, token_type_ids
        # else:
        #     label = np.array([example["label"]], dtype="int64")
        #     return input_ids, token_type_ids, label

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


train_ds, dev_ds, test_ds = load_dataset('dureader_robust', splits=('train', 'dev', 'test'))

for idx in range(2):
    print(train_ds[idx])
    print()
