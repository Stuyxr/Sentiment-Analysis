import torch
import numpy as np
from load_data import get_data
from sentiment_analysis import SentimentAnalysisModule

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

net = SentimentAnalysisModule(300, 1000, 2).to(device)
net.load_state_dict(torch.load('./checkpoints/sentiment_2.pt'))

# 改这处的代码
_, _, test_dataset = get_data()
input_data = torch.autograd.Variable(test_dataset).to(device)
output = net(input_data).cpu()

_, predicted = torch.max(output[-1, :, :], 1)
np.savetxt("predicted_results.csv", predicted.numpy(), delimiter=",", fmt="%d")
