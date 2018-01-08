import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

BATCH = 32
DIGITS = 11
EPOCH = 300
CLASSES = ['FizzBuzz', 'Fizz', 'Buzz', '']

CUDA = torch.cuda.is_available()

class FizzBuzz(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FizzBuzz, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, out_channel)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# Normal FizzBuzz
def fizz_buzz(num):
    if num % 15 == 0:
        return 0 # 'FizzBuzz'
    elif num % 3 == 0:
        return 1 # 'Fizz'
    elif num % 5 == 0:
        return 2 # 'Buzz'
    else:
        return 3 # ''


def encoder(num):
    return list(map(lambda x: int(x), ('{:0' + str(DIGITS) + 'b}').format(num))) 


def make_data(range_, batch_size):
    min, max = range_
    xs = []
    ys = []
    for x in range(min, max):
        xs += [encoder(x)]
        ys += [fizz_buzz(x)]

    data = []
    for b in range((max-min)//batch_size):
        xxs = xs[b*batch_size:(b+1)*batch_size]
        yys = ys[b*batch_size:(b+1)*batch_size]
        data += [(xxs, yys)]
        
    return data


def training(model, optimizer, training_data):
    model.train()

    for data, label in training_data:
        data = Variable(torch.FloatTensor(data))
        label = Variable(torch.LongTensor(label))

        if CUDA:
            data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        out = model(data)
        classification_loss = F.cross_entropy(out, label)
        classification_loss.backward()
        optimizer.step()


def testing(model, data):
    model.eval()
    loss = []
    correct = 0

    for x, y in data:
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.LongTensor(y))

        if CUDA:
            x, y = x.cuda(), y.cuda()

        out = model(x)
        loss += [F.cross_entropy(out, y).data[0]]
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()

    avg_loss = sum(loss) / len(loss)

    return {
        'accuracy': 100.0 * correct/(len(loss)*BATCH),
        'avg_loss': avg_loss
    }


def interactive_test(model):
    while True:
        num = input()
        if num == 'q':
            print('Bye~')
            return
        else:
            num = int(num)

        ans = fizz_buzz(num)
        x = Variable(torch.FloatTensor([encoder(num)]))
        if CUDA:
            x = x.cuda()

        predict = model(x).data.max(1, keepdim=True)[1]
        print('Predict: {}, Real_Answer: {}'.format(CLASSES[predict[0][0]], CLASSES[ans]))


if __name__ == '__main__':
    m = FizzBuzz(DIGITS, 4)
    if CUDA:
        m = m.cuda()

    optimizer = optim.SGD(m.parameters(), lr=0.02, momentum=0.9)

    print('Making dataset...')
    training_data = make_data((0, 2000), BATCH)
    testing_data = make_data((1000, 2000), BATCH)

    print('Training...')
    for epoch in range(1, EPOCH + 1):
        training(m, optimizer, training_data)
        res = testing(m, testing_data)
        print('Epoch {}, Loss: {:.5f}, Accuracy: {:.4f}%'.format(
                epoch,
                res['avg_loss'],
                res['accuracy'],
            ))

    print('Inertactive test...')
    print('Enter a digit smaller than 2^{}. ("q" to quit)'.format(DIGITS))
    interactive_test(m)

