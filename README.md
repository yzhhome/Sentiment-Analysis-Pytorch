## 中文情感分析的Pytorch实现
## 使用的Pytorch版本是1.8

**本项目使用了word2vec的中文预训练向量

**模型分别有双向LSTM加attention和普通的双向LSTM两种，自行选择

## 使用说明：
1、项目配置参数在SA_Config.py文件中，一般无需改动

2、运行SA_DataProcess.py，生成相应的word2id，word2vec等文件

3、运行SA_Train.py，得到验证精度最好的模型，并保存模型

4、运行SA_Test.py，加载模型，并用测试集进行测试

5、运行SA_Predict.py，加载模型，并进行预测

## 跑完30个epoch最好的验证精度在93.8%
## 模型的准确率平均在85%左右
