"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        import gdown
        gdown.download(id="1D9pJHM2Pod_7UGwAFQeQb2Ak3nBzF6gY", output=classifier_path, quiet=False)
        gdown.download(id="18RBjRC47WCR8f6c1UAVr5Ntiq9Apcbg6", output=localizer_path, quiet=False)
        gdown.download(id="1b5uhnTZWLFqlIC1ywtJPP9QJP87_voM2", output=unet_path, quiet=False)
        
        self.classifier_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer_model = VGG11Localizer(in_channels=in_channels)
        self.segmenter_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def load_weights(model, path):
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu")
                model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

        load_weights(self.classifier_model, classifier_path)
        load_weights(self.localizer_model, localizer_path)
        load_weights(self.segmenter_model, unet_path)

        self.classifier_head = self.classifier_model.classifier
        self.bbox_head = self.localizer_model.regressor

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        e1 = self.segmenter_model.enc1(x)
        p1 = self.segmenter_model.pool1(e1)

        e2 = self.segmenter_model.enc2(p1)
        p2 = self.segmenter_model.pool2(e2)

        e3 = self.segmenter_model.enc3(p2)
        p3 = self.segmenter_model.pool3(e3)

        e4 = self.segmenter_model.enc4(p3)
        p4 = self.segmenter_model.pool4(e4)

        e5 = self.segmenter_model.enc5(p4)
        p5 = self.segmenter_model.pool5(e5)

        shared_features = p5

        logits = self.classifier_head(shared_features)
        bbox = self.bbox_head(shared_features)

        c = self.segmenter_model.center(p5)

        d5 = self.segmenter_model.up5(c)
        d5 = torch.cat((e5, d5), dim=1)
        d5 = self.segmenter_model.dec5(d5)

        d4 = self.segmenter_model.up4(d5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.segmenter_model.dec4(d4)

        d3 = self.segmenter_model.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.segmenter_model.dec3(d3)

        d2 = self.segmenter_model.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.segmenter_model.dec2(d2)

        d1 = self.segmenter_model.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.segmenter_model.dec1(d1)

        seg_mask = self.segmenter_model.final_conv(d1)

        return {
            'classification': logits,
            'localization': bbox,
            'segmentation': seg_mask
        }
