from pointwise_matching.data_module import PointwiseMatchingDataModule
from pointwise_matching.model import PointwiseMatchingModel
from lib.trainer import Trainer
from lib.metric_stategy import AccuracyMetricStrategy, MetricStrategy
from lib.optimizer_stategy import BaseOptimizerStrategy
from sentiment_analysis.data_module import SentimentAnalysisDataModule
from sentiment_analysis.model import SentimentAnalysisModel

epochs = 10

# 数据
data_module = SentimentAnalysisDataModule()
# 模型
model = SentimentAnalysisModel(num_classes=data_module.num_classes())
# 优化器策略
optimizer_strategy = BaseOptimizerStrategy(model, data_module, epochs)
# 评价指标
metric_strategy: MetricStrategy = AccuracyMetricStrategy()

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
