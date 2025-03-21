import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4
context_length = 16 
learning_rate = 1e-3
max_iters = 5000
eval_iters = 50
device = ('mps' if torch.backends.mps.is_available() 
else ('cuda' if torch.cuda.is_available() else 'cpu'))
# TODO(rogerluo): 这里使用 mps device，loss会不断变大不收敛
device = 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)
print("device:", device)

# load file
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()
print("raw text:\n", text[:100])
print("raw text len:", len(text))

# Tokenize the text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
print("tokenized text:\n", tokenized_text[:100])
max_token_value = max(tokenized_text)
print("max token value:", max_token_value)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

# Split the tokenized text into train and test
train_size = int(0.9 * len(tokenized_text))
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]


class MultiHeadAttention(nn.Module):
    def __init__(self, context_length, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # [batch_size, context_length, d_model]
        H = self.num_heads

        # 计算 Q、K、V 并重塑多头形式，将num_heads维度从第三维移动到第二维，使得不同注意力头可以并行计算
        q = self.query(x).view(B, T, H, self.head_size).transpose(1, 2) # [batch_size, num_heads, context_length, head_size]
        k = self.key(x).view(B, T, H, self.head_size).transpose(1, 2) # [batch_size, num_heads, context_length, head_size]
        v = self.value(x).view(B, T, H, self.head_size).transpose(1, 2) # [batch_size, num_heads, context_length, head_size]

        # 计算注意力权重
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1])) # [batch_size, num_heads, context_length, context_length]
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [batch_size, num_heads, context_length, context_length]
        scores = F.softmax(scores, dim=-1) # [batch_size, num_heads, context_length, context_length]    

        # 计算输出并重塑回原始维度。在transpose之后如果要进行view操作，需要先调用contiguous令张量连续
        output = scores @ v # [batch_size, num_heads, context_length, head_size]
        output = output.transpose(1, 2).contiguous().view(B, T, C) # [batch_size, context_length, d_model]
        output = self.projection(output)
        output = self.dropout(output)
        return output
    

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.fnn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.fnn(x)


class TransformerBlock(nn.Module):
    def __init__(self, context_length, d_model, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(context_length, d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # 和原始论文不同，新的transformer block，会先进行 layer_norm
        x = x + self.attention(self.layer_norm1(x)) # [batch_size, context_length, d_model]
        x = x + self.feed_forward(self.layer_norm2(x)) # [batch_size, context_length, d_model]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = torch.zeros((1, max_len, d_model)).to(device)
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /
             torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.positional_encoding[:, :, 0::2] = torch.sin(X)
        self.positional_encoding[:, :, 1::2] = torch.cos(X)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.shape[1], :]
        return self.dropout(x)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_heads, num_blocks, dropout):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, context_length)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(context_length, d_model, num_heads, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.shape # [batch_size, context_length]
        x = self.positional_encoding(self.token_embedding(x))
        x = self.transformer_blocks(x)
        logits = self.output_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: [batch_size, context_length]
        for _ in range(max_new_tokens):
            # 截取最后 context_length 个token作为上下文
            idx_crop = idx[:, -self.context_length:]
            logits, loss = self.forward(idx_crop)
            # 只需要最后一个位置的预测结果
            logits_last_token = logits[:, -1, :]
            probs = F.softmax(logits_last_token, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True) 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 初始化模型
model = TransformerLanguageModel(vocab_size=max_token_value + 1, context_length=16, d_model=64, num_heads=4, num_blocks=8, dropout=0.1).to(device)


def get_batch(split):
    data = train_data if split == 'train' else valid_data
    # 随机生成 batch_size 个起始索引
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # 对于每个起始索引，提取长度为 context_length 的序列作为输入 x
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device) # [batch_size, context_length]
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device) # [batch_size, context_length]
    return x, y


@torch.no_grad() # 不计算梯度
def estimate_loss():
    out = {}
    model.eval() # 切换到评估模式
    for split in ['train', 'valid']:
        # 创建一个张量存储 eval_iters 次评估的损失
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # 计算平均损失
    model.train() # 切换回训练模式
    return out  # 返回一个字典，包含训练集和验证集的平均损失 {'train': loss, 'valid': loss}


# 训练模型
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_loss = list()
# 训练循环，总共训练 max_iters 步
for step in range(max_iters):
    # 每 eval_iters 步或最后一步评估一次模型性能
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_loss.append(losses)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
    # 训练步骤
    xb, yb = get_batch('train') # 获取一个训练批次
    optimizer.zero_grad(set_to_none=True)  # 清除之前的梯度
    logits, loss = model(xb, yb) # 前向传播，计算损失
    loss.backward() # 反向传播，计算新梯度
    optimizer.step() # 更新模型参数

# 保存模型
torch.save(model.state_dict(), 'model-ckpt.pt')

# 模型推理
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
y = model.generate(x, max_new_tokens=100)   # 使用模型生成新的文本，最多生成 100 个新 token
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
