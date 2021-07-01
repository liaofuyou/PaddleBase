import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import ErnieGramModel


class PointwiseMatchingModel(nn.Layer):

    def __init__(self, dropout=None):
        super().__init__()

        self.ptm = ErnieGramModel.from_pretrained('ernie-gram-zh')

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

        # 损失函数
        self.criterion = nn.loss.CrossEntropyLoss()

    def forward(self, input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # ernie
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        # dropout
        cls_embedding = self.dropout(cls_embedding)
        # 下游
        logits = F.softmax(self.classifier(cls_embedding))

        return logits

    def training_step(self, batch):
        # 参数们
        input_ids, token_type_ids, labels = batch

        # 过模型
        logits = self.forward(input_ids=input_ids, token_type_ids=token_type_ids)

        # 算loss
        loss = self.criterion(logits, labels)

        # 返回
        return loss, logits

    def validation_step(self, batch):
        return self.training_step(batch)
