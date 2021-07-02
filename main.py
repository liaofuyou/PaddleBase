from lib.metric_stategy import MetricStrategy, ChunkMetricStrategy
from lib.optimizer_stategy import BaseOptimizerStrategy
from lib.trainer import Trainer
from ner.data_module import NerDataModule
from ner.model import NERInformationExtraction

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

# controller = PointwiseMatchingController()
# controller.test()


# data = {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}
# data2 = {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}
# 
# controller.predict([data, data2])
