import torch
from model import Model

device = ('mps' if torch.backends.mps.is_available() 
else ('cuda' if torch.cuda.is_available() else 'cpu'))
print("device:", device)
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)

# 和训练的流程一样，重复构建同样的词表
# 加载文本
with open('data/scifi.all', 'r', encoding='utf-8') as file:
    text = file.read()
print(f"文本长度为: {len(text)}")
print(text[:100])

# 构建词汇表 
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(f"词汇表大小为: {vocab_size}")
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}
encode = lambda x: [char2idx[char] for char in x]
decode = lambda idxs: ''.join([idx2char[idx] for idx in idxs])
print(encode("hello world"))
print(decode(encode("hello world")))

# 加载模型
model = Model(vocab_size).to(device)
print("loading model...")
#model.load_state_dict(torch.load('model/model-scifi.pt'))#
model.load_state_dict(torch.load('model/model-scifi-finetune.pt'))
print("model loaded")
model.eval()

# 推理
start = "小明喜欢打篮球，体育很不错，他立志要成为球星。"
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
y = model.generate(x, max_new_tokens=500)
print(decode(y[0].tolist()))
