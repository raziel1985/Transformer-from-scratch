import json
import os
import torch
from datetime import datetime
from model import Model

batch_size = 8
context_length = 128
max_iters = 1000
learning_rate = 1e-4 
eval_interval = 50
eval_iters = 10
device = ('mps' if torch.backends.mps.is_available() 
else ('cuda' if torch.cuda.is_available() else 'cpu'))
print("device:", device)
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# 加载finetune文本
with open('data_finetune/scifi-finetune.json', 'r', encoding='utf-8') as file:
    alpaca = json.load(file)
    text = alpaca[1000:5000]
print("finetune data num:", len(text))
print("finetune data sample:", text[0])
# TODO(rogerluo): 这里对finetune数据源进行暴力改造(json -> string)，只是为了简化快速跑通demo
# 正常情况下应该先将text按行为单位，划分到训练和验证集；且每行数据都需要终结符；并在预测时按照finetune的数据格式拼接prompt
# 当前情况将所有数据都拼接到一起了，finetune训练集合中的prompt格式被破坏了
text = str(text)

use_mini_vocab = True
if use_mini_vocab:
    # 复用训练时的小词表，需要通过加载训练时的文本来构建词表
    with open('data/scifi.all', 'r', encoding='utf-8') as file:
        text = file.read()
    print(f"文本长度为: {len(text)}")
    print(text[:100])
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    print(f"词汇表大小为: {vocab_size}")
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    # TODO(rogerluo): 将词表中没有出现过的词，强制mapping到最后一个token idx上
    encode = lambda x: [char2idx[char] if char in char2idx else vocab_size - 1 for char in x]
    decode = lambda idxs: ''.join([idx2char[idx] for idx in idxs])
    print(encode("hello world"))
    print(decode(encode("hello world")))
    tokenized_text = torch.tensor(encode(text), dtype=torch.long, device=device)
else:
    # 使用cl100k_base通用词表
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    print(f"词表大小为: {vocab_size}")
    print(tokenizer.encode("hello world"))
    print(tokenizer.decode(tokenizer.encode("hello world")))
    tokenized_text = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

print(f"tokenized_text shape: {tokenized_text.shape}")
print(tokenized_text[:100])
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# 加载模型
model = Model(vocab_size).to(device)
model.load_state_dict(torch.load('model/model-scifi.pt'))


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[i:i + context_length] for i in idxs]).to(device)
    y = torch.stack([data[i + 1:i + context_length + 1] for i in idxs]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# finetune模型
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {current_time}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 保存最终的模型
torch.save(model.state_dict(), 'model/model-scifi-finetune.pt')
