import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

from eval import evaluate
from model import TextClassificationModel, OneLayerFFNN, TwoLayerFFNN, ThreeLayerFFNN
from predict import predict
from train import train
from utils import yield_tokens, text_pipeline, collate_batch

# 1 : World 2 : Sports 3 : Business 4 : Sci/Tec
train_iter = AG_NEWS(split="train")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64  # Here, you can modify the code to change the embedding size.

# Model. You can change the model to TextClassificationModel, OneLayerFFNN, TwoLayerFFNN, ThreeLayerFFNN
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# Hyperparameters

EPOCHS = 50  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "Adagrad": torch.optim.Adagrad,
    "RMSProp": torch.optim.RMSprop
}
OPTIMIZER = "Adam"  # optimizer. You can choose SGD, Adam, Adagrad, RMSProp

print(f"Training on {device}.")
print(
    f"---Hyperparameters---\n"
    f"Epochs: {EPOCHS}, LR: {LR}, Batch size: {BATCH_SIZE}, Optimizer: {OPTIMIZER}\n"
    f"---------------------")

criterion = torch.nn.CrossEntropyLoss()
optimizer = OPTIMIZERS[OPTIMIZER](model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(model, train_dataloader, criterion, optimizer, epoch)
    accu_val = evaluate(model, valid_dataloader, criterion)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)

print(f"valid accuracy: {total_accu:8.3f}")
print("Checking the results of test dataset.")
print(f"Epoch: {EPOCHS}, LR: {LR}, Batch size: {BATCH_SIZE}, Optimizer: {OPTIMIZER}")
accu_test = evaluate(model, test_dataloader, criterion)
print(f"test accuracy: {accu_test:8.3f}")

ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

print("This is a %s news" % ag_news_label[predict(model, ex_text_str, text_pipeline)])
