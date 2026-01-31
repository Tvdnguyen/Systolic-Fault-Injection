"""
Traffic Sign Recognition Network - Small Tensor Version
Designed to reduce tensor sizes in early layers to avoid OOM
Target: >90% accuracy with smaller memory footprint
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficSignNetSmall(nn.Module):
    """
    Smaller version of TrafficSignNet with reduced channel counts

    Key changes from original:
    - conv1: 32 -> 16 channels (reduces first tensor from 32x32x32 to 16x32x32)
    - conv2: 32 -> 16 channels
    - conv3: 64 -> 32 channels (reduces tensor from 64x16x16 to 32x16x16)
    - conv4: 64 -> 32 channels
    - conv5: 128 -> 64 channels (reduces tensor from 128x8x8 to 64x8x8)
    - conv6: 128 -> 64 channels
    - fc1: 512 -> 256 features
    - fc2: 256 -> 128 features

    Expected tensor sizes:
    - After conv1: [1, 16, 32, 32] = 16,384 elements (vs 32,768 original)
    - After conv2: [1, 16, 32, 32] = 16,384 elements (vs 32,768 original)
    - After conv3: [1, 32, 16, 16] = 8,192 elements (vs 16,384 original)
    - After conv4: [1, 32, 16, 16] = 8,192 elements
    - After conv5: [1, 64, 8, 8] = 4,096 elements (vs 8,192 original)
    - After conv6: [1, 64, 8, 8] = 4,096 elements

    Total memory reduction: ~50% in early layers
    """

    def __init__(self, num_classes=43):
        super(TrafficSignNetSmall, self).__init__()

        # First conv block: 16 channels (reduced from 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Second conv block: 32 channels (reduced from 64)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Third conv block: 64 channels (reduced from 128)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4

        # Fully connected layers
        # After pool3: 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # Reduced from 512
        self.fc2 = nn.Linear(256, 128)  # Reduced from 256
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def get_layer_info(self):
        """Return list of (layer_name, layer_type, layer_module, output_shape_str)"""
        return [
            ('conv1', 'Conv', self.conv1, '(1, 16, 32, 32)'),
            ('conv2', 'Conv', self.conv2, '(1, 16, 32, 32)'),
            ('conv3', 'Conv', self.conv3, '(1, 32, 16, 16)'),
            ('conv4', 'Conv', self.conv4, '(1, 32, 16, 16)'),
            ('conv5', 'Conv', self.conv5, '(1, 64, 8, 8)'),
            ('conv6', 'Conv', self.conv6, '(1, 64, 8, 8)'),
            ('fc1', 'Linear', self.fc1, '(1, 256)'),
            ('fc2', 'Linear', self.fc2, '(1, 128)'),
            ('fc3', 'Linear', self.fc3, '(1, 43)'),
        ]


class TrafficSignNetSmallNoDropout(nn.Module):
    """
    No-dropout version for verification (auto_LiRPA compatibility)
    """

    def __init__(self, num_classes=43):
        super(TrafficSignNetSmallNoDropout, self).__init__()

        # First conv block: 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv block: 32 channels
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third conv block: 64 channels
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def load_from_dropout_checkpoint(self, checkpoint_path):
        """Load weights from TrafficSignNetSmall checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load state dict (skip dropout layers)
        state_dict = checkpoint['model_state_dict']

        # Filter out dropout weights if any
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'dropout' not in k}

        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")

    def get_layer_info(self):
        """Return list of (layer_name, layer_type, layer_module, output_shape_str)"""
        return [
            ('conv1', 'Conv', self.conv1, '(1, 16, 32, 32)'),
            ('conv2', 'Conv', self.conv2, '(1, 16, 32, 32)'),
            ('conv3', 'Conv', self.conv3, '(1, 32, 16, 16)'),
            ('conv4', 'Conv', self.conv4, '(1, 32, 16, 16)'),
            ('conv5', 'Conv', self.conv5, '(1, 64, 8, 8)'),
            ('conv6', 'Conv', self.conv6, '(1, 64, 8, 8)'),
            ('fc1', 'Linear', self.fc1, '(1, 256)'),
            ('fc2', 'Linear', self.fc2, '(1, 128)'),
            ('fc3', 'Linear', self.fc3, '(1, 43)'),
        ]


if __name__ == '__main__':
    # Test model
    model = TrafficSignNetSmall(num_classes=43)
    x = torch.randn(1, 3, 32, 32)

    print("TrafficSignNetSmall Architecture:")
    print(model)

    print("\n" + "="*80)
    print("Testing forward pass...")
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*80)
    print("Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test no-dropout version
    print("\n" + "="*80)
    print("Testing TrafficSignNetSmallNoDropout...")
    model_nodrop = TrafficSignNetSmallNoDropout(num_classes=43)
    output_nodrop = model_nodrop(x)
    print(f"Output shape: {output_nodrop.shape}")

    print("\n✓ All tests passed!")
