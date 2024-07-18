import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalCNN(nn.Module):
    def __init__(self):
        super(MultimodalCNN, self).__init__()

        # Image processing layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.pool2 = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(512 * 7 * 7, 1536)
        self.fc2 = nn.Linear(1536, 128)
        
        # Metadata processing layers
        self.meta_fc1 = nn.Linear(15, 64)
        self.meta_fc2 = nn.Linear(64, 128)
        self.meta_fc3 = nn.Linear(128, 128)
        
        # Combined layers
        self.comb_fc1 = nn.Linear(256, 128)
        self.comb_fc2 = nn.Linear(128, 64)
        self.comb_fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, meta):
        # Image processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = self.pool(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Metadata processing
        meta = F.relu(self.meta_fc1(meta))
        meta = F.relu(self.meta_fc2(meta))
        meta = self.meta_fc3(meta)
        
        # Combine image and metadata features
        combined = torch.cat((x, meta), dim=1)
        combined = F.relu(self.comb_fc1(combined))
        combined = self.dropout(combined)
        combined = F.relu(self.comb_fc2(combined))
        combined = self.dropout(combined)
        combined = self.comb_fc3(combined)
        
        return combined

# Example instantiation
model = MultimodalCNN()
