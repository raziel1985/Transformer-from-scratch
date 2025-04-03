import math
import torch
import torch.nn as nn
from torch.nn import functional as F

context_length = 128
d_model = 512
num_blocks = 12
num_heads = 8
dropout = 0.1

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.fnn(x)


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
        # 使用下三角掩码，使得每个(query)位置只能看到它之前的(key)位置，实现因果性。
        # 这样做的目的是，在生成文本时，每个位置只能看到它之前生成的内容，而不能看到之后生成的内容，从而实现因果性。
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [batch_size, num_heads, context_length, context_length]
        scores = F.softmax(scores, dim=-1) # [batch_size, num_heads, context_length, context_length]    

        # 计算输出并重塑回原始维度。在transpose之后如果要进行view操作，需要先调用contiguous令张量连续
        output = scores @ v # [batch_size, num_heads, context_length, head_size]
        output = output.transpose(1, 2).contiguous().view(B, T, C) # [batch_size, context_length, d_model]
        output = self.projection(output)
        output = self.dropout(output)
        return output


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
        self.positional_encoding = torch.zeros((1, max_len, d_model))
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /
             torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))  # [max_len, d_model]
        # 偶数维度使用sin，奇数维度使用cos
        self.positional_encoding[:, :, 0::2] = torch.sin(X) # [max_len, d_model // 2]
        self.positional_encoding[:, :, 1::2] = torch.cos(X) # [max_len, d_model // 2]

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, context_length)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(context_length, d_model, num_heads, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(d_model, vocab_size) 
        
    def forward(self, x, targets=None):
        B, T = x.shape # [batch_size, context_length]
        x = self.positional_encoding(self.token_embedding(x))
        x = self.transformer_blocks(x)
        logits = self.output_layer(x) # [batch_size, context_length, vocab_size]

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx: [batch_size, context_length]
        for _ in range(max_new_tokens):
            # 截取最后 context_length 个token作为上下文
            idx_crop = idx[:, -context_length:] # [batch_size, context_length]
            # TODO(rogerluo): 这里的forward每次会对所有context token进行预测，效率很低, 可以使用cache优化
            logits, loss = self.forward(idx_crop)
            # 只需要最后一个位置的预测结果
            # temperature 越大，概率分布越分散，越随机
            # temperature 越小，概率分布越集中，越确定
            logits_last_token = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            probs = F.softmax(logits_last_token, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
            idx = torch.cat((idx, idx_next), dim=1) # [batch_size, context_length + 1]
        return idx


# cpu训练日志，mps的速度会快一倍
# step 50: train loss 5.9306, val loss 5.8353, time 2025-03-22 01:46:26
# step 100: train loss 5.5684, val loss 5.4013, time 2025-03-22 01:47:28
# step 150: train loss 5.2893, val loss 5.1732, time 2025-03-22 01:48:30
# step 200: train loss 5.1900, val loss 5.1472, time 2025-03-22 01:49:35
# step 250: train loss 5.0646, val loss 5.0446, time 2025-03-22 01:50:40
# step 300: train loss 5.0220, val loss 4.9377, time 2025-03-22 01:51:45
# step 350: train loss 5.0568, val loss 4.9033, time 2025-03-22 01:52:50
# step 400: train loss 4.9005, val loss 4.8551, time 2025-03-22 01:53:57
# step 450: train loss 4.8140, val loss 4.7848, time 2025-03-22 01:55:03
# step 500: train loss 4.8200, val loss 4.7636, time 2025-03-22 01:56:14
# step 600: train loss 4.7447, val loss 4.7127, time 2025-03-22 01:58:37
# step 700: train loss 4.7732, val loss 4.6082, time 2025-03-22 02:00:57
# step 800: train loss 4.6962, val loss 4.6699, time 2025-03-22 02:03:18
# step 900: train loss 4.6462, val loss 4.6121, time 2025-03-22 02:05:40
# step 1000: train loss 4.5859, val loss 4.6097, time 2025-03-22 02:08:04
# step 2000: train loss 4.4524, val loss 4.3602, time 2025-03-22 02:32:17
# step 3000: train loss 4.3406, val loss 4.2358, time 2025-03-22 02:56:14
# step 5000: train loss 4.1092, val loss 4.1200, time 2025-03-22 03:43:34
# step 10000: train loss 3.9168, val loss 3.8791, time 2025-03-22 05:41:32
# step 15000: train loss 3.7666, val loss 3.7367, time 2025-03-22 07:39:01
# step 19000: train loss 3.6583, val loss 3.6944, time 2025-03-22 09:14:32
# step 19999: train loss 3.6881, val loss 3.6516, time 2025-03-22 09:37:56
