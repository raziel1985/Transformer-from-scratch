import tiktoken
import torch
from datetime import datetime
from model import Model

batch_size = 12
context_length = 128
max_iters = 20000
learning_rate = 1e-3
eval_interval = 50
eval_iters = 20
device = ('mps' if torch.backends.mps.is_available() 
else ('cuda' if torch.cuda.is_available() else 'cpu'))
print("device:", device)
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


# 加载文本
with open('data/scifi.all', 'r', encoding='utf-8') as file:
    text = file.read()
print(f"文本长度为: {len(text)}")
print(text[:100])

# 加载词表，对文本进行编码
use_mini_vocab = True
if use_mini_vocab:
    # 构建最小化的词汇表，仅使用数据集中出现过的单词构建最小化的词表，词表大约是7K多
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    print(f"词汇表大小为: {vocab_size}")
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    encode = lambda x: [char2idx[char] for char in x]
    decode = lambda idxs: ''.join([idx2char[idx] for idx in idxs])  
    print(encode("hello world"))
    print(decode(encode("hello world")))
    tokenized_text = torch.tensor(encode(text), dtype=torch.long, device=device)
else:
    # 使用cl100k_base，通用性更好，词表大约是100K多，但是embadding层更大训练速度更慢，模型也更大
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


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[i:i + context_length] for i in idxs]).to(device)
    y = torch.stack([data[i + 1:i + context_length + 1] for i in idxs]).to(device)
    return x, y


# 构建模型
model = Model(vocab_size).to(device)


# 计算 loss
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

# 训练模型
print("start training")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {current_time}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # 每1000轮保存一次模型
    if step % 1000 == 0:
        model_file = 'model-scifi-' + str(int(step / 1000)) + '.pt'
        print(f"start to save model: {model_file}")
        torch.save(model.state_dict(), 'model/' + model_file)
        print(f"finish saving model")

# 保存最终的模型
torch.save(model.state_dict(), 'model/model-scifi.pt')
