import paddle.nn as nn
from paddlenlp.transformers import ErnieForTokenClassification

from lib.metric_stategy import ChunkMetricStrategy
from lib.optimizer_stategy import BaseOptimizerStrategy
from lib.trainer import Trainer
from ner.data_module import NerDataModule


class NERInformationExtraction(nn.Layer):

    def __init__(self, num_classes):
        super().__init__()

        self.ptm = ErnieForTokenClassification.from_pretrained('ernie-1.0', num_classes=num_classes)
        self.criterion = nn.loss.CrossEntropyLoss(ignore_index=-1)

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
        return loss, logits

    def validation_step(self, batch):
        return self.training_step(batch)

    @staticmethod
    def run():
        epochs = 10
        # 数据
        data_module = NerDataModule()
        # 模型
        model = NERInformationExtraction(num_classes=data_module.num_classes())
        # 优化策略
        optimizer_strategy = BaseOptimizerStrategy(model, data_module, epochs)
        # 评估策略
        metric_strategy = ChunkMetricStrategy(label_list=data_module.label_vocab.keys())

        trainer = Trainer(data_module,
                          model,
                          optimizer_strategy,
                          metric_strategy,
                          epochs)

        trainer.train()
