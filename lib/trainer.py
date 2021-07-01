import os

import numpy as np
import paddle

from lib.metric_stategy import MetricStrategy
from lib.optimizer_stategy import OptimizerStrategy


class Trainer:

    def __init__(self,
                 data_module,
                 model,
                 optimizer_strategy: OptimizerStrategy,
                 metric_strategy: MetricStrategy,
                 epochs=10):

        self.epochs = epochs
        self.global_step = 0

        # 模型
        self.data_module = data_module
        self.model = model
        self.metric_strategy = metric_strategy

        # 优化器策略
        self.lr_scheduler, self.optimizer = optimizer_strategy.get_scheduler_and_optimizer()

    def train(self):
        """训练"""

        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(self.data_module.train_dataloader, start=1):

                self.global_step += 1

                # 训练
                loss, logits = self.model.training_step(batch)

                # 评价指标
                self.metric_strategy.compute_train_metric(
                    loss, logits, epoch, self.global_step, step, batch)

                # 反向传播
                self.backward(loss)

                # 验证集上进行评估
                if self.global_step % 100 == 0:
                    self.evaluate()

        self.save_model()

    @paddle.no_grad()
    def evaluate(self, dev=True):
        """验证/评估"""

        # 进入 eval 模式
        self.model.eval()
        self.metric_strategy.get_metric().reset()

        if dev:
            tag = "dev"
            dataloader = self.data_module.dev_dataloader
        else:
            tag = "Test"
            dataloader = self.data_module.train_dataloader

        # 在验证集/测试上跑一遍
        acc = 0
        losses = []
        for batch in dataloader:
            loss, logits = self.model.validation_step(batch)

            # 评价指标
            acc = self.metric_strategy.compute_dev_metric(logits, batch)
            losses.append(loss.numpy())

        print(tag + " evaluate loss: %.5f, accu: %.5f" % (np.mean(losses), acc))

        # 进入 train 模式
        if dev:
            self.model.train()

        # 重置
        self.metric_strategy.get_metric().reset()

    @paddle.no_grad()
    def test(self):
        """测试"""
        self.evaluate(dev=False)

    @paddle.no_grad()
    def predict(self, lines: list):
        """预测"""
        batch_probs = []

        self.model.eval()

        with paddle.no_grad():
            for batch_data in self.data_module.predict_dataloader(lines):
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
                batch_prob = self.model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()

                batch_probs.append(batch_prob)

            batch_probs = np.concatenate(batch_probs, axis=0)

            return batch_probs

    def save_model(self, dir_name="ckpt"):

        # 训练结束后，存储模型参数
        save_dir = os.path.join(dir_name, "model_%d" % self.global_step)
        os.makedirs(save_dir)

        # 保存当前模型参数
        paddle.save(self.model.state_dict(), os.path.join(save_dir, 'model_state.pdparams'))
        # 保存tokenizer的词表等
        self.data_module.tokenizer.save_pretrained(save_dir)

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()
