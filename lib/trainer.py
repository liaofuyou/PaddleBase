import os

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
            for step, batch_data in enumerate(self.data_module.train_dataloader, start=1):

                self.global_step += 1
                progress = (epoch, self.global_step, step)

                # 训练
                model_ret = self.model.training_step(batch_data)

                # 评估指标
                self.metric_strategy.compute_train_metric(batch_data, model_ret, progress)

                # 反向传播
                self.backward(model_ret["loss"])

                # 验证集上进行评估
                if self.global_step % 100 == 0:
                    self.evaluate()

        self.save_model()

    @paddle.no_grad()
    def evaluate(self, dev=True):
        """验证/评估"""

        # 进入 eval 模式
        self.model.eval()
        self.metric_strategy.reset()

        if dev:
            dataloader = self.data_module.dev_dataloader
        else:
            dataloader = self.data_module.train_dataloader

        # 在验证集/测试上跑一遍
        batch_count = len(dataloader)
        for step, batch_data in enumerate(dataloader, start=1):
            model_ret = self.model.validation_step(batch_data)
            # 评估指标
            self.metric_strategy.compute_dev_metric(batch_data, model_ret, batch_count == step)

        # 进入 train 模式
        if dev:
            self.model.train()

        # 重置
        self.metric_strategy.reset()

    @paddle.no_grad()
    def test(self):
        """测试"""
        self.evaluate(dev=False)

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
