from ner.controller import NerController
from pointwise_matching.controller import PointwiseMatchingController
from sentiment_analysis.controller import SentimentAnalysisController

controller = PointwiseMatchingController()
# controller.test()


data = {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}
data2 = {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}

controller.predict([data, data2])
