import paddle

from pointwise_matching.data_module import PointwiseMatchingDataModule
from pointwise_matching.model import PointwiseMatchingModel


class PointwiseMatchingPredictor:

    @paddle.no_grad()
    def predict(self, data_list: list):
        """预测"""

        data_module = PointwiseMatchingDataModule(is_predict=True)
        model = PointwiseMatchingModel()
        model.eval()

        # 预测
        batch_probs = []
        for batch_data in data_module.predict_dataloader(data_list):
            input_ids, token_type_ids = batch_data

            batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob[0][0])

        # 构造结果对象
        for data, probs in zip(data_list, batch_probs):
            data["value"] = probs

        return data_list


text = [{'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'},
        {'query': '喜欢打篮球9的男生喜欢什么==样的女生', 'title': '爱打篮球的男生喜欢什么样的女生'}]

p = PointwiseMatchingPredictor().predict(text)
print(p)
