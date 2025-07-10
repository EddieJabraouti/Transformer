import torch 
import torch.nn as nn 
from torch.nn import functional as F

print(torch.__version__)

batch_size = 32 
block_size = 8
max_iters = 3000
eval_interval = 300 
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r') as f:
    text = f.read()

#all the unique characters in the data set
chars = sorted(list(set(text))) #creates the text into a set and then makes a sorted list out of each character alphabetically.
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#create a mapping for the characters to ints

stoi = {ch: i for i,ch in enumerate(chars)} #individualy maps each character to an integer for all characters in chars
itos = {i:ch for i,ch in enumerate(chars)} #individual maps each integer to its designated characters for all ints in chars
encode = lambda s: [stoi[c] for c in s] #encoder takes a string and ouputs it as a list of integers
decoder = lambda l: ''.join([itos[i] for i in l]) #decoder takes a list of integers and outputs a string


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) #first 90% of the data set will be for training, the rest for validation
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  #generate a small batch of data for inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix]) #torch.stack takes each consecutive chunk of the data set and concatenates it along a new dimension
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #set model to eval mode

    #Estimate loss separately for training and validation sets
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        #For each of eval_iters iterations, sample a batch and compute the loss to average over multiple evaluations
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out #resets back to training mode


class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #creates an embedding table full of raw scores correlating to the probability of a word. Each row is a word that has a corresponding vector to it

  def forward(self, idx, targets=None):
      logits = self.token_embedding_table(idx) #logits will correlate the the vector at the idx

      if targets is None:
        loss = None
      else:
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        #measure the quality of predictions
        loss = F.cross_entropy(logits, targets)   #given we have the target how well have we predicted it based on logits


      return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      #get the predictions
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      #sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for iter in range(max_iters):

    #evaluate the loss on the train and val sets every once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:4f}")


    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


print(decoder(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))