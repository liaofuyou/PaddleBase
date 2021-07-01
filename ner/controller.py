import os

import numpy as np
import paddle

from ner.data_module import DataModule
from lib.metric_stategy import MetricStrategy, ChunkMetricStrategy
from ner.model import NERInformationExtraction
from lib.optimizer_stategy import OptimizerStrategy, BaseOptimizerStrategy


class NerController:

    def __init__(self):

        self.epochs = 10
        self.global_step = 0
        ptm_name = 'ernie-1.0'

        # 数据
        self.data_module = DataModule(pretrained_model=ptm_name)

        # 模型
        self.model = NERInformationExtraction(pretrained_model=ptm_name,
                                              num_classes=self.data_module.num_classes())

        # 优化器策略
        optimizer_strategy: OptimizerStrategy = BaseOptimizerStrategy(self.model, self.data_module, self.epochs)
        self.lr_scheduler, self.optimizer = optimizer_strategy.get_scheduler_and_optimizer()

        # 评价指标
        self.metric_strategy: MetricStrategy = ChunkMetricStrategy(label_list=self.data_module.label_vocab.keys())

    def train(self):
        """训练"""

        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(self.data_module.train_dataloader, start=1):
                input_ids, token_type_ids, lens, labels = batch

                self.global_step += 1

                # 训练
                loss, logits = self.model.training_step(batch)

                # 评价指标
                self.metric_strategy.compute_train_metric(
                    logits, labels, lens, epoch, self.global_step, step, loss)

                # 反向传播
                self.backward(loss)

                # 验证集上进行评估
                if self.global_step % 100 == 0:
                    self.evaluate()

        self.save_model()

    @paddle.no_grad()
    def evaluate(self):
        """验证/评估"""

        # 进入 eval 模式
        self.model.eval()
        self.metric_strategy.get_metric().reset()

        # 在验证集上跑一遍
        for batch in self.data_module.dev_dataloader:
            input_ids, token_type_ids, lens, labels = batch

            logits = self.model.validation_step(batch)

            # 评价指标
            self.metric_strategy.compute_dev_metric(logits, labels, lens)

        # 进入 train 模式
        self.model.train()
        self.metric_strategy.get_metric().reset()

    @staticmethod
    @paddle.no_grad()
    def predict(model, data_loader):
        """预测"""

        batch_probs = []

        # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
        model.eval()

        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
                batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()

                batch_probs.append(batch_prob)
            batch_probs = np.concatenate(batch_probs, axis=0)

            return batch_probs

    def save_model(self):

        # 训练结束后，存储模型参数
        save_dir = os.path.join("checkpoint", "model_%d" % self.global_step)
        os.makedirs(save_dir)

        # 保存参数
        save_param_path = os.path.join(save_dir, 'model_state.pdparams')
        paddle.save(self.model.state_dict(), save_param_path)

        # 保存 tokenize
        self.data_module.tokenizer.save_pretrained(save_dir)

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()
