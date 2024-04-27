import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from process import tokenize, stem, bag_of_words
from data_model import NeuralNet
import multiprocessing 


if __name__ == '__main__':
    multiprocessing.freeze_support()

    with open('data.json', 'r') as datafile:
        data = json.load(datafile)

    all_words = []
    tags = []
    xy = []

    for datavalue in data['data']:
        tag = datavalue['tag']
        tags.append(tag)
        for pattern in datavalue['pattern']:
            word = tokenize(pattern)
            all_words.extend(word)
            xy.append((word, tag))

    ignore_words = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '[', ']', '{', '}', ';', "'", '\\', '"', '|', ',', '.', '<', '>', '/', '?', '`', '~']
    all_words = [stem(word) for word in all_words if word not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    Y_train = []

    for (word, tag) in xy:
        bag = bag_of_words(word, all_words)
        X_train.append(bag)
        

        label = tags.index(tag)
        Y_train.append(label)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = Y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words_ofword, labels) in train_loader:
            words = words_ofword.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, loss ={loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')

datas = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(datas, FILE)

print(f'training complete. file saved to {FILE}')

print(data)   
