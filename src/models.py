import torch.nn as nn
from transformers import RobertaPreTrainedModel

class CustomCNNLayer(nn):
    def __init__(self):
        super(CustomCNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

class RobertaWithCNN(RobertaPreTrainedModel):
    pass