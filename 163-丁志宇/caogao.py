#
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

'''

基于pytorch的网络编写
实现一个网络完成简单的nlp任务
输入一个字符串，根据字符所在位置进行分类


'''


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)    # +1的原因是可能出现a不存在的情况
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]    # 或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)   # 这里用sample是不放回的采样，每个字母不会重复出现，但要求字符串长度小于词表长度
    if "a" in x:
        y = x.index("a")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    print("本次预测集共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 20    # 训练轮数
    batch_size = 40     # 每次训练样本的个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 30         # 每个字的维度
    sentence_length = 10   # 样本文本长度
    learning_rate = 0.005   # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()     # 梯度归零
            loss = model(x, y)    # 计算loss
            loss.backward()       # 计算梯度
            optim.step()          # 更新权重
            watch_loss.append(loss.item())
        print("========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [a[0] for a in log], label="acc")
    plt.plot(range(len(log)), [a[1] for a in log], label="loss")
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model2.pth")
    # 保存词表
    writer = open("vocab2.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    
    return 


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 10
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))


if __name__ == "__main__":
    main()
    test_strings = ["gijkbcdeaf", "kijhdefacb", "kijabcdefh", "gkijadfbec"]
    predict("model2.pth", "vocab2.json", test_strings)
