import abc
import time

import paddle
from paddle.metric import Accuracy
from paddlenlp.metrics import ChunkEvaluator


class MetricStrategy:
    """评估策略"""

    @abc.abstractmethod
    def get_metric(self):
        pass

    @abc.abstractmethod
    def compute_train_metric(self, loss, logits, epoch, global_step, step, batch):
        pass

    @abc.abstractmethod
    def compute_dev_metric(self, logits, batch):
        pass


class ChunkMetricStrategy(MetricStrategy):
    """Chunk评估策略"""

    def __init__(self, label_list):
        super().__init__()
        self.tic_train = time.time()
        self.metric = ChunkEvaluator(label_list=label_list, suffix=True)

    def get_metric(self):
        return self.metric

    def _compute(self, preds, labels, lens):
        """计算评价指标，并返回对应的值"""
        n_infer, n_label, n_correct = self.metric.compute(None, lens, preds, labels)
        self.metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = self.metric.accumulate()
        return precision, recall, f1_score

    def compute_train_metric(self, loss, logits, epoch, global_step, step, batch):
        """计算训练评价指标"""
        _, _, lens, labels = batch

        preds = paddle.argmax(logits, axis=-1)
        precision, recall, f1_score = self._compute(preds, labels, lens)

        # 每间隔 10 step 输出训练指标
        # if global_step % 2 != 0:
        #     return

        # 打印日志
        text = "global step %d, epoch: %d, batch: %d, loss: %.5f, precision: %f - recall: %f - f1: %f, speed: %.2f step/s" % (
            global_step, epoch, step, loss, precision, recall, f1_score, 10 / (time.time() - self.tic_train))
        print(text)

        # 更新时间
        self.tic_train = time.time()

    def compute_dev_metric(self, logits, batch):
        """计算验证评价指标"""
        _, _, lens, labels = batch

        preds = paddle.argmax(logits, axis=-1)
        precision, recall, f1_score = self._compute(preds, labels, lens)
        print("dev precision: %f - recall: %f - f1: %f" % (precision, recall, f1_score))


class AccuracyMetricStrategy(MetricStrategy):
    """Accuracy评估策略"""

    def __init__(self):
        super().__init__()
        self.tic_train = time.time()
        self.metric: Accuracy = Accuracy()

    def get_metric(self):
        return self.metric

    def _compute(self, preds, labels):
        """计算评价指标，并返回对应的值"""

        correct = self.metric.compute(preds, labels)
        self.metric.update(correct)
        return self.metric.accumulate()

    def compute_train_metric(self, loss, logits, epoch, global_step, step, batch):
        """计算训练评价指标"""

        # 一般来说， 元组的最后一个元素都是 label
        labels = batch[-1]

        preds = paddle.argmax(logits, axis=-1)
        acc = self._compute(preds, labels)

        # 每间隔 10 step 输出训练指标
        # if global_step % 2 != 0:
        #     return

        # 打印日志
        text = "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s" % (
            global_step, epoch, step, loss, acc, 10 / (time.time() - self.tic_train))
        print(text)

        # 更新时间
        self.tic_train = time.time()

    def compute_dev_metric(self, logits, batch):
        """计算验证评价指标"""

        # 一般来说， 元组的最后一个元素都是 label
        labels = batch[-1]

        preds = paddle.argmax(logits, axis=-1)
        return self._compute(preds, labels)
