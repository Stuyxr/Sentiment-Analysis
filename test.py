import torch
from load_data import get_data
from sentiment_analysis import SentimentAnalysisModule

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

net = SentimentAnalysisModule(300, 1000, 2).to(device)
net.load_state_dict(torch.load('./checkpoints/sentiment_2.pt'))

# 改这处的代码
'''_, _, test_dataset = get_data()
input_data = torch.autograd.Variable(test_dataset).to(device)
output = net(input_data).cpu()'''

while(True):
    test_sentence = input("请输入一个句子：")
    _, _, test_dataset = get_data(True, True, test_sentence)
    input_data = torch.autograd.Variable(test_dataset).to(device)
    print("input_data值")
    print(input_data)
    output = net(input_data).cpu()
    print("output值")
    print(output)
    _, predicted = torch.max(output[-1, :, :], 1)
    print(predicted)

'''
_, predicted = torch.max(output[-1, :, :], 1)
with open('1170300418.csv', 'a') as f:
    for i in range(len(predicted)):
        f.write(f'{i}, {predicted[i]}\r\n')'''
