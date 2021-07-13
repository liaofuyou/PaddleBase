import abc
import time

import numpy as np
import paddle
from paddle.metric import Accuracy
from paddlenlp.metrics import ChunkEvaluator


class MetricStrategy:
    """评估策略"""

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def compute_train_metric(self, batch_data, model_ret, progress):
        pass

    @abc.abstractmethod
    def compute_dev_metric(self, batch_data, model_ret, final_step=False):
        pass


class ChunkMetricStrategy(MetricStrategy):
    """Chunk评估策略"""

    def __init__(self, label_list):
        super().__init__()
        self.tic_train = time.time()
        self.metric = ChunkEvaluator(label_list=label_list, suffix=True)

    def reset(self):
        return self.metric.reset()

    def _compute(self, preds, labels, lens):
        """计算评估指标，并返回对应的值"""
        n_infer, n_label, n_correct = self.metric.compute(None, lens, preds, labels)
        self.metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = self.metric.accumulate()
        return precision, recall, f1_score

    def compute_train_metric(self, batch_data, model_ret, progress):
        """计算训练评估指标"""
        _, _, lens, labels = batch_data

        epoch, global_step, step = progress
        loss = model_ret["loss"]
        logits = model_ret["logits"]

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

    def compute_dev_metric(self, batch_data, model_ret, final_step=False):
        """计算验证评估指标"""
        _, _, lens, labels = batch_data
        logits = model_ret["logits"]

        preds = paddle.argmax(logits, axis=-1)
        precision, recall, f1_score = self._compute(preds, labels, lens)
        print("dev precision: %f - recall: %f - f1: %f" % (precision, recall, f1_score))


class AccuracyMetricStrategy(MetricStrategy):
    """Accuracy评估策略"""

    def __init__(self):
        super().__init__()
        self.tic_train = time.time()
        self.metric: Accuracy = Accuracy()
        self.losses = []

    def reset(self):
        return self.metric.reset()

    def _compute(self, preds, labels):
        """计算评估指标，并返回对应的值"""

        correct = self.metric.compute(preds, labels)
        self.metric.update(correct)
        return self.metric.accumulate()

    def compute_train_metric(self, batch_data, model_ret, progress):
        """计算训练评估指标"""

        # 一般来说， 元组的最后一个元素都是 label
        labels = batch_data[-1]
        epoch, global_step, step = progress
        loss = model_ret["loss"]
        logits = model_ret["logits"]

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

    def compute_dev_metric(self, batch_data, model_ret, final_step=False):
        """计算验证评估指标"""

        # 一般来说， 元组的最后一个元素都是 label
        labels = batch_data[-1]
        logits = model_ret["logits"]
        loss = model_ret["loss"]

        self.losses.append(loss)

        preds = paddle.argmax(logits, axis=-1)
        acc = self._compute(preds, labels)

        if final_step:
            print("evaluate loss: %.5f, accu: %.5f" % (np.mean(self.losses), acc))


class TranslateMetricStrategy(MetricStrategy):
    """机器翻译评估策略"""

    def __init__(self):
        super().__init__()
        self.total_sum_cost = 0
        self.total_token_num = 0
        self.tic_train = time.time()

    def reset(self):
        self.total_sum_cost = 0
        self.total_token_num = 0
        pass

    def _compute(self, sum_cost, token_num):
        """计算评估指标，并返回对应的值"""
        self.total_sum_cost += sum_cost.numpy()
        self.total_token_num += token_num.numpy()
        total_avg_cost = self.total_sum_cost / self.total_token_num

        perplexity = np.exp([min(total_avg_cost, 100)])

        return total_avg_cost, perplexity

    def compute_train_metric(self, batch_data, model_ret, progress):
        """计算训练评估指标"""

        epoch, global_step, step = progress
        sum_cost = model_ret["sum_cost"]
        avg_cost = model_ret["avg_cost"]
        token_num = model_ret["token_num"]

        total_avg_cost, perplexity = self._compute(sum_cost, token_num)

        # 每间隔 10 step 输出训练指标
        # if global_step % 2 != 0:
        #     return

        print("step_idx: %d, epoch: %d, batch: %d, avg loss: %f, ppl: %f " %
              (global_step, epoch, step, total_avg_cost, perplexity))

    def compute_dev_metric(self, batch_data, model_ret, final_step=False):
        """计算验证评估指标"""
        sum_cost = model_ret["sum_cost"]
        avg_cost = model_ret["avg_cost"]
        token_num = model_ret["token_num"]

        total_avg_cost, perplexity = self._compute(sum_cost, token_num)

        if final_step:
            print("validation, avg loss: %f,  ppl: %f" % (total_avg_cost, perplexity))
