from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size =32
block_size = 8
max_iters = 1000
eval_interval =300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200

torch.manual_seed(1337)

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)

stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join([itos[i] for i in l])

#train vs test split
data = torch.tensor(encoder(text), dtype=torch.long)
n = len(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loader for inputs and targets
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    
    x= torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    model.eval() then to model.train()
    good practice to know mode your neural network.
    efficient in memory use, we dont have to call backward() propagation
    """
    out = {}
    model.eval() #eval 
    for split in ['train', 'test']:
        loss = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits,loss = model(X,Y)
            loss[k] = loss.item()
        out[split]=loss.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super.__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets: torch.tensor |None=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            
            loss = F.cross_entroy(logits, targets)
        return loss, logits

    def generate(self,idx, max_tokens):
        for _ in range(max_tokens):
            loss,logits= self(idx)
            
            lgotis = logits[:,-1:]
            prob = F.softmax(logits, dim =-1)
            idx_next = torch.multinomial(prob, num_samples =1)
            
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# pytorch AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

for iter in range(max_iters):
    if iter % eval_interval ==0:
        loss = estimate_loss()
        print(f"step:{iter}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")
    
    xb,yb = get_batch('train')
    
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    optimizer.step()

# generating from model 
context = torch.zeros((1,1), dtype = torch.long, device =device)
print(decoder(m.generate(context,max_tokens = 500)[0].tolist()))       