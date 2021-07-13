import paddle

from sentiment_analysis.data_module import SentimentAnalysisDataModule
from sentiment_analysis.model import SentimentAnalysisModel


class SentimentAnalysisPredictor:

    @paddle.no_grad()
    def predict(self, data_list: list):
        """预测"""

        data_module = SentimentAnalysisDataModule(is_predict=True)
        model = SentimentAnalysisModel()
        model.eval()

        # 预测
        batch_probs = []
        for batch_data in data_module.predict_dataloader(data_list):
            input_ids, token_type_ids, qid = batch_data

            batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob[0][0])

        # 构造结果对象
        for data, probs in zip(data_list, batch_probs):
            data["value"] = probs

        return data_list


text = [{'text': '喜欢打篮球的男生喜欢什么样的女生', 'qid': 1},
        {'text': '喜欢打篮球9的男生喜欢什么==样的女生', 'qid': 2}]

p = SentimentAnalysisPredictor().predict(text)
print(p)
