import paddle
import tqdm
from ner.data_module import NerDataModule
from ner.model import NERInformationExtraction


class NerPredictor:

    @paddle.no_grad()
    def predict(self, data_list: list):
        """预测"""

        data_module = NerDataModule(is_predict=True)
        model = NERInformationExtraction(num_classes=data_module.num_classes())
        model.eval()

        pred_list = []
        len_list = []
        # 预测
        batch_probs = []
        for batch_data in data_module.predict_dataloader(data_list):
            input_ids, token_type_ids, lens = batch_data
            logits = model(input_ids, token_type_ids)
            pred = paddle.argmax(logits, axis=-1)
            pred_list.append(pred.numpy())
            len_list.append(lens.numpy())

        preds = self.parse_decodes(data_module.predict_ds, pred_list, len_list, data_module.label_vocab)

        return preds

    @staticmethod
    def parse_decodes(ds, decodes, lens, label_vocab):
        decodes = [x for batch in decodes for x in batch]
        lens = [x for batch in lens for x in batch]
        id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

        outputs = []
        for idx, end in enumerate(lens):
            sent = ds.data[idx][:end]
            tags = [id_label[x] for x in decodes[idx][1:end]]

            sent_out = []
            tags_out = []
            words = ""
            for s, t in zip(sent, tags):
                print("====1", s)
                print("====2", t)
                if t.endswith('-B') or t == 'O':
                    if len(words):
                        sent_out.append(words)
                    tags_out.append(t.split('-')[0])
                    words = s
                else:
                    words += s
            if len(sent_out) < len(tags_out):
                sent_out.append(words)
            outputs.append(''.join(
                [str((s, t)) for s, t in zip(sent_out, tags_out)]))
        return outputs


text = [
    "黑龙江省双鸭山市尖山区八马路与东平行路交叉口北40米韦业涛18600009172",
]

p = NerPredictor().predict(text)
print(p)
