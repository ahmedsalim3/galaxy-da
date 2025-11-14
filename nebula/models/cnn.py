import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=3, input_size=(3, 28, 28)):
        super().__init__()

        self.features = nn.Sequential(
            # first conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size
        _dummy_input = torch.zeros(1, *input_size)
        with torch.no_grad():
            out = self.features(_dummy_input)
        self.flatten_dim = out.view(1, -1).size(1)

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self.class_scales = nn.Parameter(torch.ones(num_classes))

    def forward(self, x):
        x = self.features(x)
        z = x.view(x.size(0), -1)
        out = self.classifier(z)
        out = out * self.class_scales.unsqueeze(0)
        return out, z


if __name__ == "__main__":
    # Example usage with any input size
    model = CNN(input_size=(3, 64, 64))
    x = torch.randn(8, 3, 64, 64)
    out, features = model(x)
    print(out.shape, features.shape)
