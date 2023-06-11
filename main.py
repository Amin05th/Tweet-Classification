import pandas as pd
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing_utils import stemming, ignore_words, tokenization, bag_of_words

writer = SummaryWriter()

# configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# configure batch size
batch_size = 8

# load data
train_df = pd.read_csv("../nlp-getting-started/train.csv")
test_df = pd.read_csv("../nlp-getting-started/test.csv")
submission_df = pd.read_csv("../nlp-getting-started/sample_submission.csv")


# clean data and split data
train_df.drop(["id", "keyword", "location"], inplace=True, axis=1)
test_df.drop(["id", "keyword", "location"], inplace=True, axis=1)


# preprocess data
def preprocess_data(text_list):
    preprocessed_sentences = []
    for text in text_list:
        text = tokenization(text)
        text = ignore_words(text)
        text = stemming(text)
        preprocessed_sentences.append(text)
    return bag_of_words(preprocessed_sentences)


X_train = preprocess_data(train_df["text"].to_numpy())
y_train = train_df["target"].to_numpy()
X_test = test_df["text"].to_numpy()


# create dataset
class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_samples = len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def __len__(self):
        return self.n_samples


train_dataset, val_dataset = random_split(TweetDataset(X_train, y_train), [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# train
def train(epoch, model, optimizer, criterion):
    running_loss = 0.0
    for epoch in range(epoch):
        for idx, (data, target) in enumerate(train_dataloader):
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            if (idx + 1) % 100 == 0:
                print(
                    f'Train Epoch: {epoch} [{idx * len(data)}/{len(train_dataset)} ({100. * idx / len(train_dataset):.0f}%)]\tLoss: {loss.item():.6f}')
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_dataloader) + idx)
                running_loss = 0.0


# test
def test(model, criterion):
    with torch.no_grad():
        correct = 0
        loss = 0
        for data, target in val_dataloader:
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            out = model(data)
            loss += criterion(out, target).item()
            _, predict = torch.max(out.data, 1)
            correct += (target == predict).sum().item()
        print("Mean Loss ", loss / len(val_dataloader.dataset))
        print("Correct in % ", 100. / len(val_dataset) * correct)


writer.close()
