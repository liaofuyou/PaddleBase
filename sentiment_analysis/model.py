import paddle
import paddle.nn as nn
from paddlenlp.transformers import SkepForSequenceClassification

from lib.metric_stategy import AccuracyMetricStrategy
from lib.optimizer_stategy import BaseOptimizerStrategy
from lib.trainer import Trainer
from sentiment_analysis.data_module import SentimentAnalysisDataModule


class SentimentAnalysisModel(nn.Layer):

    def __init__(self):
        super().__init__()
        # Define the model network and its loss
        self.ptm = SkepForSequenceClassification.from_pretrained("skep_ernie_1.0_large_ch")

        # 采用交叉熵 损失函数
        self.criterion = nn.loss.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None):
        return self.ptm(input_ids, token_type_ids)

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
        data_module = SentimentAnalysisDataModule()
        # 模型
        model = SentimentAnalysisModel()
        # 优化器策略
        optimizer_strategy = BaseOptimizerStrategy(model, data_module, epochs)
        # 评价指标
        metric_strategy = AccuracyMetricStrategy()

        trainer = Trainer(data_module,
                          model,
                          optimizer_strategy,
                          metric_strategy,
                          epochs)

        trainer.train()
