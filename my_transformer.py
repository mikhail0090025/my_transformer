import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class EmbeddingsLayer(nn.Module):
    def __init__(self, embeddings_size, vocabulary_size):
        super(EmbeddingsLayer, self).__init__()
        embeddings = torch.zeros((vocabulary_size, embeddings_size), dtype=torch.float32)
        nn.init.xavier_uniform_(embeddings[1:])
        self.embeddings = nn.Parameter(embeddings)

    def forward(self, x):
        # x — тензор индексов токенов (batch_size, seq_length)
        mask = (x != 0).float()
        return self.embeddings[x], mask

class SelfAttention(nn.Module):
    def __init__(self, embeddings_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embeddings_size, embeddings_size)
        self.key = nn.Linear(embeddings_size, embeddings_size)
        self.value = nn.Linear(embeddings_size, embeddings_size)
        self.scale = embeddings_size ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x
        batch_size, seq_length, embeddings_size = x.size()
        q = self.query(x)  # (batch_size, seq_length, embeddings_size)
        k = self.key(x)    # (batch_size, seq_length, embeddings_size)
        v = self.value(x)  # (batch_size, seq_length, embeddings_size)

        # Вычисление внимания: QK^T / sqrt(d_k)
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (batch_size, seq_length, seq_length)
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_length, seq_length)
        output = torch.bmm(attention_weights, v)  # (batch_size, seq_length, embeddings_size)

        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embeddings_size, heads_count=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.all_sa = nn.ModuleList([SelfAttention(embeddings_size) for _ in range(heads_count)])
        self.final_layer = nn.Linear(embeddings_size * heads_count, embeddings_size)
        self.embeddings_size = embeddings_size
        self.heads_count = heads_count
    
    def forward(self, x):
        # (batch_size, embeddings count, embeddings size)
        x, mask = x
        batch_size = x.shape[0]
        mask = mask.bool()
        result = []
        full_size = x.shape[1]
        for i, current_embeddings in enumerate(x):
            current_mask = mask[i]
            real_embeddings = current_embeddings[current_mask]
            initial_embeddings = real_embeddings.clone()
            all_heads = torch.tensor(np.array([]), dtype=torch.float32)
            for head in self.all_sa:
                attentioned = head(real_embeddings.unsqueeze(0))
                all_heads = torch.cat((all_heads, head(attentioned)), dim=2)
            output = self.final_layer(all_heads)
            output += initial_embeddings
            padding = torch.zeros((1, full_size - output.shape[1], self.embeddings_size))
            output = torch.cat((output, padding), dim=1)
            result.append(output.squeeze(0))
        if len(result) == 1:
            output = result[0].unsqueeze(0)
        else:
            output = torch.stack(result, dim=0)
        
        return output

'''
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embeddings_size, heads_count=3):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embeddings_size, heads_count)
        self.final_layer = nn.Linear(embeddings_size, embeddings_size)

    def forward(self, x):
        embeddings, mask = x
        print("Embeddings: ", embeddings.shape)
        print("Mask: ", mask.shape)
        batch_size, seq_length, embeddings_size = embeddings.size()

        # Переставляем размерности: (seq_length, batch_size, embeddings_size)
        embeddings = embeddings.transpose(0, 1)
        # attn_mask = (mask == 0).transpose(0, 1)
        attn_mask = (mask == 0)
        print("Attention mask: ", attn_mask.shape)

        # Вычисляем внимание
        output, _ = self.attention(embeddings, embeddings, embeddings, key_padding_mask=attn_mask)
        output = output.transpose(0, 1)  # Обратно к (batch_size, seq_length, embeddings_size)
        output = self.final_layer(output) + embeddings.transpose(0, 1)  # Residual connection
        print("Output: ", output.shape)

        return output
'''

class MyTransformer(nn.Module):
    def __init__(self, embeddings_size, vocabulary_size, heads_count=3):
        super(MyTransformer, self).__init__()
        self.embeddings_layer = EmbeddingsLayer(embeddings_size, vocabulary_size)
        self.multihead_attention = MultiHeadSelfAttention(embeddings_size, heads_count)
        self.norm = nn.LayerNorm(embeddings_size, eps=1e-6)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(embeddings_size, 20)  # embeddings_size вместо 10000
    
    def forward(self, x):
        x = self.embeddings_layer(x)
        x = self.multihead_attention(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Среднее по последовательности
        x = self.dropout(x)
        x = self.linear(x)
        return x  # Логиты для CrossEntropyLoss
    
from transformers import BertTokenizer
from sklearn.datasets import fetch_20newsgroups

def split_into_sentences(text, min_words_between_dots=1):
    split_by_dot = str(text).split(".")
    new_list = []
    try:
        for i1 in range(0, len(split_by_dot) - 2, 3):
            i2 = i1 + 1
            i3 = i1 + 2
            l1 = len(split_by_dot[i1].strip().split(" "))
            l2 = len(split_by_dot[i2].strip().split(" "))
            l3 = len(split_by_dot[i3].strip().split(" "))
            print(l1, l2, l3)
            if l1 <= min_words_between_dots or l2 <= min_words_between_dots or l3 <= min_words_between_dots:
                new_list.append(
                    split_by_dot[i1] + ("." if l1 <= min_words_between_dots else " ") +
                    split_by_dot[i2] + ("." if l2 <= min_words_between_dots else " ") +
                    split_by_dot[i3] + ("." if l3 <= min_words_between_dots else " ")
                )
            else:
                new_list.append(split_by_dot[i1].strip())
                new_list.append(split_by_dot[i2].strip())
                new_list.append(split_by_dot[i3].strip())
    except IndexError as e:
        print("End of list")
    return new_list

print(split_into_sentences("I suck big penises. U.K. penis. Short. Not too fucking short."))

# import dataset2
# outputs = dataset2.outputs
# texts = dataset2.texts

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = len(tokenizer.vocab)
news_dataset = fetch_20newsgroups(subset="all")
texts = news_dataset.data
targets = news_dataset.target
sentenses = []
targets_new = []
all_train_losses = []
all_val_losses = []
for text, target in zip(texts[:1000], targets[:1000]):
    sentenses_ = split_into_sentences(text)
    sentenses.extend(sentenses_)
    targets_new.extend([target] * len(sentenses_))
print(targets_new)
print(len(sentenses))
print(len(targets_new))
target_names = news_dataset.target_names

class MyDataset(Dataset):
    def __init__(self, targets, texts, max_length=-1, classes_count=20):
        self.titles = np.array(texts)
        self.outputs = torch.tensor(targets, dtype=torch.long)  # Индексы классов
        self.tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentense)) for sentense in self.titles]
        print(sum([len(g) for g in self.tokens]) / len(self.tokens))
        if max_length == -1:
            max_length = max([len(token) for token in self.tokens])
        print("Max length: ", max_length)
        for i in range(len(self.tokens)):
            current_length = len(self.tokens[i])
            self.tokens[i] = self.tokens[i] + [0 for _ in range(max_length - current_length)]
        print("Dataset size: ", self.__len__())
        print("Outputs: ", self.outputs.shape)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens_list = torch.tensor(self.tokens[idx], dtype=torch.int32)
        return tokens_list, self.outputs[idx]
    
train_texts, val_texts, train_outputs, val_outputs = train_test_split(sentenses, targets_new, test_size=0.15, random_state=52)
train_dataset = MyDataset(train_outputs, train_texts)
val_dataset = MyDataset(val_outputs, val_texts)
train_dataset_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataset_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
loss_func = nn.CrossEntropyLoss()

def one_epoch(model, train_dataset_loader, val_dataset_loader, optimizer):
    model.train()
    train_loss = 0
    val_loss = 0
    for tokens_list, outputs in train_dataset_loader:
        optimizer.zero_grad()
        predicted = model(tokens_list)
        loss = loss_func(predicted, outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    for tokens_list, outputs in val_dataset_loader:
        predicted = model(tokens_list)
        loss = loss_func(predicted, outputs)
        val_loss += loss.item()

    train_loss = train_loss / len(train_dataset_loader)
    val_loss = val_loss / len(val_dataset_loader)
    print("Train loss: ", train_loss, ", Validation loss: ", val_loss)
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    return train_loss, val_loss

def go_epochs(epochs, model, train_dataset_loader, val_dataset_loader, optimizer):
    for epoch in range(epochs - 1):
        one_epoch(model, train_dataset_loader, val_dataset_loader, optimizer)
    return one_epoch(model, train_dataset_loader, val_dataset_loader, optimizer)
# text = "I suck big penises with gachi muchi guys sregsgr"
# tokens = tokenizer.tokenize(text)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(tokens)  # ['i', 'love', 'cats']
# print(ids)     # [1045, 2293, 4012]

my_trans = MyTransformer(129, vocab_size, 3)
optimizer = optim.Adam(my_trans.parameters(), lr=0.0005, weight_decay=0.0001)
output = my_trans(torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize("I suck big penises with gachi guys"))]))
