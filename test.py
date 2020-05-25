import torch
from load_data import get_data
from sentiment_analysis import SentimentAnalysisModule

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

net = SentimentAnalysisModule(300, 1000, 2).to(device)
net.load_state_dict(torch.load('./checkpoints/sentiment_2.pt'))

batch = 32

_, _, test_dataset = get_data()
input_data = torch.autograd.Variable(test_dataset).to(device)
output = net(input_data).cpu()

_, predicted = torch.max(output[-1, :, :], 1)
with open('1170300418.csv', 'a') as f:
    for i in range(len(predicted)):
        f.write(f'{i}, {predicted[i]}\r\n')
