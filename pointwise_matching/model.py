import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import ErnieGramModel

from lib.metric_stategy import AccuracyMetricStrategy
from lib.optimizer_stategy import BaseOptimizerStrategy
from lib.trainer import Trainer
from pointwise_matching.data_module import PointwiseMatchingDataModule


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
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch):
        return self.training_step(batch)

    @staticmethod
    def run():
        epochs = 10

        # 数据
        data_module = PointwiseMatchingDataModule()
        # 模型
        model = PointwiseMatchingModel()
        # 优化器策略
        optimizer_strategy = BaseOptimizerStrategy(model, data_module, epochs)
        # 评估指标
        metric_strategy = AccuracyMetricStrategy()

        trainer = Trainer(data_module,
                          model,
                          optimizer_strategy,
                          metric_strategy,
                          epochs)

        trainer.train()

        # data = {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}
        # data2 = {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}
        #
        # controller.predict([data, data2])
