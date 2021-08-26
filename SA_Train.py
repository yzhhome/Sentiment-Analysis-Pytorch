from io import open
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
from SA_DataProcess import prepare_train_data, build_word2vec, Data_set
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from SA_Model import LSTMModel, LSTM_attention
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

id2word = {}
for key, value in word2id.items():
    id2word[value] = key

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 转换句子id表示和标签
train_array, train_lable, \
val_array, val_lable = prepare_train_data(word2id,
                                    Config.train_path,
                                    Config.val_path,
                                    Config.max_sen_len)

train_dataset = Data_set(train_array, train_lable)
train_dataloader = DataLoader(train_dataset,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=0)

valid_dataset = Data_set(val_array, val_lable)
valid_dataloader = DataLoader(valid_dataset,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=0)

word2vec = build_word2vec(Config.pre_word2vec_path, word2id)
word2vec = torch.from_numpy(word2vec)
word2vec = word2vec.float()

model = LSTM_attention(vocab_size=Config.vocab_size,
                        embedding_dim=Config.embedding_dim,
                        pretrained_weight=word2vec,
                        update_w2v=Config.update_w2v,
                        hidden_dim=Config.hidden_dim,
                        num_layers=Config.num_layers,
                        drop_keep_prob=Config.drop_keep_prob,
                        n_class=Config.n_class,
                        bidirectional=Config.bidirectional)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=Config.lr)

# 情感分类属于分类任务使用交叉熵损失
criterion = nn.CrossEntropyLoss()

# 学习率调整
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

# 最好的验证精度初始化为0
best_accuracy = 0.0

epoches = Config.n_epoch

for epoch in range(epoches):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    train_dataloader = tqdm.tqdm(train_dataloader)
    train_dataloader.set_description('[%s%02d/%02d %s%f]' 
        %('Epoch:', epoch + 1, epoches, 
        'lr:', scheduler.get_last_lr()[0]))

    for step, data in (enumerate(train_dataloader)):
        optimizer.zero_grad()
        
        input, target = data[0], data[1]
        input = input.type(torch.LongTensor)
        target = target.type(torch.LongTensor)
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        #返回每行的最大值和最大值索引
        value, predicted = torch.max(output, 1)

        # 预测的总数
        total += target.size(0)

        # 预测正确的数量
        correct += (predicted == target).sum().item()

        F1 = f1_score(target.cpu(), predicted.cpu(), average='weighted')
        Recall = recall_score(target.cpu(), predicted.cpu(), average='micro')
        CM = confusion_matrix(target.cpu(), predicted.cpu())

        postfix = {'train loss: {:.5f}, train accuracy: {:.3f}%, '
                    'F1: {:.3f}%, Recall: {:.3f}%'
                    .format(train_loss / (step + 1), (correct / total) * 100, 
                    F1 * 100, Recall * 100)}
        
        train_dataloader.set_postfix_str(s=postfix)

    model.eval()

    # 评估模式不需要更新梯度
    with torch.no_grad():
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        valid_dataloader = tqdm.tqdm(valid_dataloader)
        valid_dataloader.set_description('[%s%02d/%02d %s%f]' 
            %('Epoch:', epoch + 1, epoches, 
            'lr:', scheduler.get_last_lr()[0]))

        for step, data in enumerate(valid_dataloader):                    
            input, target = data[0], data[1]
            input = input.type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            val_loss += loss.item()
            value, predicted = torch.max(output, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
            valid_accuracy = val_correct / val_total

            F1 = f1_score(target.cpu(), predicted.cpu(), average='weighted')
            Recall = recall_score(target.cpu(), predicted.cpu(), average='micro')
            CM = confusion_matrix(target.cpu(), predicted.cpu())

            postfix = {'valid loss: {:.5f}, valid accuracy: {:.3f}%, '
                        'F1: {:.3f}%, Recall: {:.3f}%'
                        .format(val_loss / (step + 1), (val_correct / val_total) * 100, 
                        F1 * 100, Recall * 100)}
            
            valid_dataloader.set_postfix_str(s=postfix)  

            # 保存验证精度最好的模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model, Config.model_state_dict_path)
                print("\n save best valid accuracy model: {:.3f} \n".format(valid_accuracy))