import abc

from paddle.optimizer import AdamW, Optimizer, Adam
from paddle.optimizer.lr import LRScheduler, NoamDecay
from paddlenlp.transformers import LinearDecayWithWarmup


class OptimizerStrategy:
    """优化器策略"""

    @abc.abstractmethod
    def get_scheduler_and_optimizer(self) -> (Optimizer, LRScheduler):
        pass


class BaseOptimizerStrategy(OptimizerStrategy):
    """默认的优化策略"""

    def __init__(self, model, data_module, epochs):
        self.model = model
        self.num_training_steps = len(data_module.train_dataloader) * epochs

    def get_scheduler_and_optimizer(self) -> (Optimizer, LRScheduler):
        lr_scheduler = LinearDecayWithWarmup(5E-5, self.num_training_steps, 0.1)

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


class TranslateOptimizerStrategy(OptimizerStrategy):
    """机器翻译优化策略"""

    def __init__(self, model):
        self.model = model

    def get_scheduler_and_optimizer(self) -> (Optimizer, LRScheduler):
        args = self.model.args
        scheduler = NoamDecay(args.d_model,
                              args.warmup_steps,
                              args.learning_rate,
                              last_epoch=0)

        # Define optimizer
        optimizer = Adam(learning_rate=scheduler,
                         beta1=args.beta1,
                         beta2=args.beta2,
                         epsilon=float(args.eps),
                         parameters=self.model.parameters())

        return scheduler, optimizer
