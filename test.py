import torch
import torch.nn as nn

model = nn.Linear(10, 3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(16, 10)
y = torch.randint(0, 3, size=(16,))  # (16, )
logits = model(x)  # (16, 3)

loss = criterion(logits, y)
