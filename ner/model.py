import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieForTokenClassification


class NERInformationExtraction(nn.Layer):

    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        # Define the model network and its loss
        self.ptm = ErnieForTokenClassification.from_pretrained(pretrained_model, num_classes=num_classes)

        # 采用交叉熵 损失函数
        self.criterion = nn.loss.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids=None):
        return self.ptm(input_ids, token_type_ids)

    def training_step(self, batch):
        input_ids, token_type_ids, length, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=token_type_ids)

        loss = paddle.mean(self.criterion(logits, labels))
        preds = paddle.argmax(logits, axis=-1)
        return loss, preds

    def validation_step(self, batch):
        input_ids, token_type_ids, length, labels = batch
        logits = self(input_ids, token_type_ids)
        preds = paddle.argmax(logits, axis=-1)
        return preds
