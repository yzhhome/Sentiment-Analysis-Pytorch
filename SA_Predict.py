from io import open
import torch
from SA_DataProcess import text_to_array_no_label
from SA_Config import Config

split_list = []
word2id = {}

# word2id转换为字典
with open(Config.word2id_path, encoding='utf-8') as f:
    for line in f.readlines():
        sp = line.strip().split()
        split_list.append(sp)
    word2id = dict(split_list)

# 字典的值转换为整型
for key in word2id:
    word2id[key] = int(word2id[key])

model = torch.load(Config.model_state_dict_path)

# 预测时在CPU上运行
model.cpu()

with torch.no_grad():
    
    # 把句子转换为id表示形式
    input_array = text_to_array_no_label(word2id, Config.max_sen_len, Config.predict_path)

    input = torch.from_numpy(input_array)
    input = input.type(torch.LongTensor)
    output = model(input)
    value, predicted = torch.max(output, 1)

    with open(Config.predict_path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            print("句子：", line, "\n", "预测类别：", predicted[i].item(), "\n")