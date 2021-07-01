import abc

from paddle.optimizer import AdamW, Optimizer
from paddle.optimizer.lr import LRScheduler
from paddlenlp.transformers import LinearDecayWithWarmup


class OptimizerStrategy:
    """优化器策略"""

    @abc.abstractmethod
    def get_scheduler_and_optimizer(self) -> (Optimizer, LRScheduler):
        pass


class BaseOptimizerStrategy(OptimizerStrategy):
    """最原始的优化器"""

    def __init__(self, model, data_module, epochs):
        self.model = model
        self.num_training_steps = len(data_module.train_dataloader) * epochs

    def get_scheduler_and_optimizer(self) -> (Optimizer, LRScheduler):
        lr_scheduler = LinearDecayWithWarmup(5E-5, self.num_training_steps, 0.0)

        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        # 定义 Optimizer
        optimizer = AdamW(
            learning_rate=lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=0.0,
            apply_decay_param_fun=lambda x: x in decay_params)

        return lr_scheduler, optimizer
