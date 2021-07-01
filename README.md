# PaddleBase

Paddle 实例项目

### 项目结构 

采用MDC设计模式（类似mvc模式）

**M = Model**：模型模块， 包括

```
class SentimentAnalysisModel(nn.Layer):
    def forward(self, batch):
    def training_step(self, batch):
    def validation_step(self, batch):
```

**D = Data Module**: 数据处理模块


```
class DataModule:
    # 第一步： 加载数据集（训练集、验证集、测试集）
    def _load_dataset():
    # 第二步： tokenize
    def _tokenize(self, max_seq_length):
    # 第三步： 构造 dataloader
    def _get_dataloader(self):
```


**C = Controller**: 控制器, 处理与客户端的交互

```
class SentimentAnalysisController:
    def train(self):
    def evaluate(self):
    def test(self):
    def save_model(self):
```


### 关于Optimizer和Metric

采用Strategy模式， 不同的情况采用不同的策略


```
class MetricStrategy:
    """评估策略"""

    @abc.abstractmethod
    def get_metric(self):
        pass

    @abc.abstractmethod
    def compute_train_metric(self, *args):
        pass

    @abc.abstractmethod
    def compute_dev_metric(self, *args):
        pass
     

class ChunkMetricStrategy(MetricStrategy):
    """Chunk评估策略"""
	。。。

class AccuracyMetricStrategy(MetricStrategy):
    """Accuracy评估策略"""
    。。。
```



```
class OptimizerStrategy:
    """优化器策略"""

    @abc.abstractmethod
    def get_scheduler_and_optimizer(self):
        pass


class BaseOptimizerStrategy(OptimizerStrategy):
    """最原始的优化器"""
	。。。
```

