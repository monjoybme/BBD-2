import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageMetadataDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = f"{self.image_folder}/{self.metadata.iloc[idx, 0]}"
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        metadata = self.metadata.iloc[idx, 1:].values.astype('float')
        sample = {'image': image, 'metadata': torch.tensor(metadata, dtype=torch.float)}

        return sample

class MultimodalCNN(nn.Module):
    def __init__(self, metadata_size):
        super(MultimodalCNN, self).__init__()
        self.image_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),    # Conv2d-1
            nn.BatchNorm2d(64),                                     # BatchNorm2d-2
            nn.ReLU(),                                              # ReLU-3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Conv2d-4
            nn.BatchNorm2d(64),                                     # BatchNorm2d-5
            nn.ReLU(),                                              # ReLU-6
            nn.MaxPool2d(kernel_size=2, stride=2),                  # MaxPool2d-7
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Conv2d-8
            nn.BatchNorm2d(128),                                    # BatchNorm2d-9
            nn.ReLU(),                                              # ReLU-10
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# Conv2d-11
            nn.BatchNorm2d(128),                                    # BatchNorm2d-12
            nn.ReLU(),                                              # ReLU-13
            nn.MaxPool2d(kernel_size=2, stride=2),                  # MaxPool2d-14
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# Conv2d-15
            nn.BatchNorm2d(256),                                    # BatchNorm2d-16
            nn.ReLU(),                                              # ReLU-17
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# Conv2d-18
            nn.BatchNorm2d(256),                                    # BatchNorm2d-19
            nn.ReLU(),                                              # ReLU-20
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# Conv2d-21
            nn.BatchNorm2d(256),                                    # BatchNorm2d-22
            nn.ReLU(),                                              # ReLU-23
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# Conv2d-24
            nn.BatchNorm2d(256),                                    # BatchNorm2d-25
            nn.ReLU(),                                              # ReLU-26
            nn.MaxPool2d(kernel_size=2, stride=2),                  # MaxPool2d-27
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),# Conv2d-28
            nn.BatchNorm2d(512),                                    # BatchNorm2d-29
            nn.ReLU(),                                              # ReLU-30
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-31
            nn.BatchNorm2d(512),                                    # BatchNorm2d-32
            nn.ReLU(),                                              # ReLU-33
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-34
            nn.BatchNorm2d(512),                                    # BatchNorm2d-35
            nn.ReLU(),                                              # ReLU-36
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-37
            nn.BatchNorm2d(512),                                    # BatchNorm2d-38
            nn.ReLU(),                                              # ReLU-39
            nn.MaxPool2d(kernel_size=2, stride=2),                  # MaxPool2d-40
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-41
            nn.BatchNorm2d(512),                                    # BatchNorm2d-42
            nn.ReLU(),                                              # ReLU-43
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-44
            nn.BatchNorm2d(512),                                    # BatchNorm2d-45
            nn.ReLU(),                                              # ReLU-46
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-47
            nn.BatchNorm2d(512),                                    # BatchNorm2d-48
            nn.ReLU(),                                              # ReLU-49
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),# Conv2d-50
            nn.BatchNorm2d(512),                                    # BatchNorm2d-51
            nn.ReLU(),                                              # ReLU-52
            nn.MaxPool2d(kernel_size=2, stride=2),                  # MaxPool2d-53
            nn.AdaptiveAvgPool2d((7, 7)),                           # AdaptiveAvgPool2d-54
        )
        
        self.image_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1536),                           # Linear-55
            nn.ReLU(),                                              # ReLU-56
            nn.Dropout(0.5),                                        # Dropout-57
            nn.Linear(1536, 128)                                    # Linear-58
        )
        
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x_image, x_metadata):
        x_image = self.image_model(x_image)
        x_image = x_image.view(x_image.size(0), -1)
        x_image = self.image_fc(x_image)

        x_metadata = self.metadata_fc(x_metadata)

        x = torch.cat((x_image, x_metadata), dim=1)
        x = self.final_fc(x)
        
        return torch.sigmoid(x)

# Example usage:
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageMetadataDataset(csv_file='path_to_csv.csv', image_folder='path_to_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

model = MultimodalCNN(metadata_size=dataset.metadata.shape[1] - 1)  # -1 to exclude image column

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(10):  # Example for 10 epochs
    for batch in dataloader:
        images, metadata = batch['image'], batch['metadata']
        outputs = model(images, metadata)
        labels = metadata[:, 0]  # Assuming the first column in metadata is the label
        loss = criterion(outputs, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
