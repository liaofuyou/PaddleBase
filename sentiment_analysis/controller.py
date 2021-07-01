import os

import paddle
import paddle.nn.functional as F

from sentiment_analysis.data_module import DataModule
from sentiment_analysis.model import SentimentAnalysisModel
from strategy.metric_stategy import MetricStrategy, AccuracyMetricStrategy
from strategy.optimizer_stategy import OptimizerStrategy, BaseOptimizerStrategy


class SentimentAnalysisController:

    def __init__(self):

        self.epochs = 10
        self.global_step = 0

        # 数据
        self.data_module = DataModule()

        # 模型
        self.model = SentimentAnalysisModel(num_classes=self.data_module.num_classes())

        # 优化器策略
        optimizer_strategy: OptimizerStrategy = BaseOptimizerStrategy(self.model, self.data_module, self.epochs)
        self.lr_scheduler, self.optimizer = optimizer_strategy.get_scheduler_and_optimizer()

        # 评价指标
        self.metric_strategy: MetricStrategy = AccuracyMetricStrategy()

    def train(self):
        """训练"""

        for epoch in range(1, self.epochs + 1):
            for step, batch in enumerate(self.data_module.train_dataloader, start=1):
                input_ids, token_type_ids, labels = batch

                self.global_step += 1

                # 训练
                loss, logits = self.model.training_step(batch)

                # 评价指标
                self.metric_strategy.compute_train_metric(
                    logits, labels, epoch, self.global_step, step, loss)

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
            input_ids, token_type_ids, labels = batch

            logits = self.model.validation_step(batch)

            # 评价指标
            self.metric_strategy.compute_dev_metric(logits, labels)

        # 进入 train 模式
        self.model.train()
        self.metric_strategy.get_metric().reset()

    @paddle.no_grad()
    def test(self):
        """预测"""

        label_map = {0: '0', 1: '1'}
        results = []
        # 切换model模型为评估模式，关闭dropout等随机因素
        self.model.eval()
        for batch in self.data_module.test_data_loader:
            input_ids, token_type_ids, qids = batch
            # 喂数据给模型
            logits = self.model(input_ids, token_type_ids)
            # 预测分类
            probs = F.softmax(logits, axis=-1)
            idx = paddle.argmax(probs, axis=1).numpy()

            idx = idx.tolist()
            labels = [label_map[i] for i in idx]
            qids = qids.numpy().tolist()
            results.extend(zip(qids, labels))
            print(zip(qids, labels))

    def save_model(self):

        # 训练结束后，存储模型参数
        save_dir = os.path.join("skep_ckpt", "model_%d" % self.global_step)
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
