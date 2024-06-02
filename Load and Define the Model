import torch
import torch.nn as nn

class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = KAN()
model.load_state_dict(torch.load('C:/Users/User/Desktop/IVF/AI/DNN/Предсказания/Prediction_KAN.pth'))
