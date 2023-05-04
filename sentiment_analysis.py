import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import get_data

class SentimentAnalysisModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentAnalysisModule, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size * 2, 500),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(500, 100),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(100, 2),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        x, _ = self.layer1(x)
        seq_len, batch_size, hidden_size = x.size()
        x = x.view(seq_len * batch_size, hidden_size)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(seq_len, batch_size, -1)
        return x

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

def calc_accuracy(net, test_set, test_target, epoch):
    correct = 0
    total = 0
    batch = 32
    with torch.no_grad():
        for i in range(0, test_set.size()[1], batch):
            input_data = torch.autograd.Variable(test_set[:, i:i+batch, :]).to(device)
            labels = Variable(test_target[i:i+batch].long())
            input_data, labels = input_data, labels
            outputs = net(input_data).cpu()
            _, predicted = torch.max(outputs[-1, :, :], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if correct > 700:
        torch.save(net.state_dict(), f'./checkpoints/sentiment_{epoch}.pt')
    print(f'accuracy: {correct}/{total}')


def main():
    train_dataset, train_label, test_dataset = get_data(True)
    print(train_dataset[:, 1, :])
    print(train_dataset[:, 1000, :])
    validation_dataset, validation_label = train_dataset[:, :1000, :], train_label[:1000]
    train_dataset, train_label = train_dataset[:, 1000:, :], train_label[1000:]
    net = SentimentAnalysisModule(300, 1000, 2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batch = 32
    for epoch in range(50):
        avg_loss = 0
        cnt = 0
        net.train()
        for i in range(0, train_dataset.size()[1], batch):
            cnt += 1
            input_data = Variable(train_dataset[:, i:i+batch, :]).to(device)
            label = Variable(train_label[i:i+batch].long()).to(device)
            output = net(input_data)[-1, :, :]
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= cnt
        print(f'epoch {epoch}: loss = {avg_loss}')
        net.eval()
        calc_accuracy(net, validation_dataset, validation_label, epoch)
        if epoch % 5 == 0 and epoch > 0:
            torch.save(net.state_dict(), f'./checkpoints/sentiment_{epoch}.pt')

if __name__ == '__main__':
    main()
