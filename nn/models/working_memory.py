import torch
import torch.nn as nn
from torchvision import models


class ResNetLSTMFeedback(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout_rate=0.3):
        super(ResNetLSTMFeedback, self).__init__()

        # Keep ResNet50 as the backbone
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Selectively unfreeze the last few layers of ResNet50
        for name, param in self.resnet.named_parameters():
            if "layer4" in name:  # Only unfreeze the last layer
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.resnet_out_features = 2048  # ResNet50 output features

        # Add layer normalization for better stability
        self.layer_norm = nn.LayerNorm(self.resnet_out_features)
        self.dropout = nn.Dropout(dropout_rate)

        # Keep original LSTM structure
        self.lstm = nn.LSTM(
            input_size=self.resnet_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # Improved feedback mechanism
        self.feedback_projection = nn.Sequential(
            nn.Linear(hidden_size, self.resnet_out_features),
            nn.LayerNorm(self.resnet_out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Learnable feedback scaling with better initialization
        self.feedback_scale = nn.Parameter(torch.zeros(1))

        # Enhanced classifier with additional layers
        self.classifier = nn.Sequential(
            nn.Linear(self.resnet_out_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        modified_resnet_features = []
        h_t = None

        for t in range(timesteps):
            # Current frame processing
            current_frame = x[:, t, :, :, :]
            resnet_features = self.resnet(current_frame)
            resnet_features = resnet_features.view(batch_size, -1)

            # Apply layer normalization and dropout
            resnet_features = self.layer_norm(resnet_features)
            resnet_features = self.dropout(resnet_features)

            # Apply feedback if available
            if h_t is not None:
                # Get last hidden state
                last_hidden = h_t[0][-1]
                feedback = self.feedback_projection(last_hidden)

                # Improved feedback scaling
                feedback_scale = (
                    torch.sigmoid(self.feedback_scale) * 0.5
                )  # Limit maximum scale
                resnet_features = resnet_features + feedback_scale * feedback

            modified_resnet_features.append(resnet_features.unsqueeze(1))

            # Update LSTM state
            _, h_t = self.lstm(resnet_features.unsqueeze(1), h_t)

        # Process all features
        all_features = torch.cat(modified_resnet_features, dim=1)
        final_features = all_features[:, -1, :]

        # Final classification
        output = self.classifier(final_features)
        return output


class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout_rate=0.5):
        super(ResNetLSTM, self).__init__()

        # Load ResNet50 architecture (without weights)
        # self.resnet = models.resnet50(weights=None)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        # Load your fine-tuned weights
        # self.resnet.load_state_dict(torch.load('version1_analysis/resnet50_finetuned_coco1400.pth'))

        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze ResNet50 layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Get the number of features from the last layer of ResNet50
        self.resnet_out_features = (
            2048  # This is the number of features output by ResNet50
        )

        self.dropout = nn.Dropout(dropout_rate)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.resnet_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Combine features
        self.combine_features = nn.Sequential(
            nn.Linear(self.resnet_out_features + hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Final classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        # Extract features using ResNet50
        resnet_features = self.resnet(c_in)
        resnet_features = resnet_features.view(batch_size, timesteps, -1)

        # Apply dropout to ResNet features
        resnet_features = self.dropout(resnet_features)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(resnet_features)
        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)

        # Get the last output of LSTM
        lstm_last_out = lstm_out[:, -1, :]

        # Get the last ResNet features
        resnet_last_out = resnet_features[:, -1, :]

        # Combine LSTM and ResNet features
        combined = torch.cat((resnet_last_out, lstm_last_out), dim=1)
        # Apply dropout to combined features
        combined = self.dropout(combined)

        combined_features = self.combine_features(combined)
        # Apply dropout before final classification
        combined_features = self.dropout(combined_features)

        # Final classification
        output = self.classifier(combined_features)

        return output
