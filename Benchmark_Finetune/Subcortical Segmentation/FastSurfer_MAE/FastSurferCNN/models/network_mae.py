# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, nn

if TYPE_CHECKING:
    import yacs.config

import FastSurferCNN.models.interpolation_layer as il
import FastSurferCNN.models.sub_module as sm
from .mae_cnn import MAE_CNN


class FastSurferCNNBase(nn.Module):
    """
    Network Definition of Fully Competitive Network network.

    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)

    Attributes
    ----------
    encode1, encode2, encode3, encode4
        Competitive Encoder Blocks.
    decode1, decode2, decode3, decode4
        Competitive Decoder Blocks.
    bottleneck
        Bottleneck Block.

    Methods
    -------
    forward
        Feedforward through graph.
    """

    def __init__(self, params: dict, padded_size: int = 256):
        """
        Construct FastSurferCNNBase object.

        Parameters
        ----------
        params : Dict
            Parameters in dictionary format

        padded_size : int, default = 256
            Size of image when padded (Default value = 256).
        """
        super().__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params["num_channels"] = params["num_filters"]
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params["num_channels"] = params["num_filters"]
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        params["num_filters_last"] = params["num_filters"]
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Tensor,
        scale_factor: Tensor | None = None,
        scale_factor_out: Tensor | None = None,
    ) -> Tensor:
        """
        Feedforward through graph.

        Parameters
        ----------
        x : Tensor
            Input image [N, C, H, W] representing the input data.
        scale_factor : Tensor, optional
            [N, 1] Defaults to None.
        scale_factor_out : Tensor, optional
            Tensor representing the scale factor for the output. Defaults to None.

        Returns
        -------
        decoder_output1 : Tensor
            Prediction logits.
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(
            encoder_output1
        )
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(
            encoder_output2
        )
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(
            encoder_output3
        )

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(
            decoder_output4, skip_encoder_3, indices_3
        )
        decoder_output2 = self.decode2.forward(
            decoder_output3, skip_encoder_2, indices_2
        )
        decoder_output1 = self.decode1.forward(
            decoder_output2, skip_encoder_1, indices_1
        )

        return decoder_output1


class FastSurferCNN(FastSurferCNNBase):
    """
    Main Fastsurfer CNN Network.

    Attributes
    ----------
    classifier
        Initialized Classification Block.

    Methods
    -------
    forward
        Feedforward through graph.
    """

    def __init__(self, params: dict, padded_size: int):
        """
        Construct FastSurferCNN object.

        Parameters
        ----------
        params : Dict
            Dictionary of configurations.
        padded_size : int
            Size of image when padded.
        """
        super().__init__(params)
        params["num_channels"] = params["num_filters"]
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Tensor,
        scale_factor: Tensor | None = None,
        scale_factor_out: Tensor | None = None,
    ) -> Tensor:
        """
        Feedforward through graph.

        Parameters
        ----------
        x : Tensor
            Input image [N, C, H, W].
        scale_factor : Tensor, optional
            [N, 1] Defaults to None.
        scale_factor_out : Tensor, optional
            Tensor representing the scale factor for the output. Defaults to None.

        Returns
        -------
        output : Tensor
            Prediction logits.
        """
        net_out = super().forward(x, scale_factor)
        output = self.classifier.forward(net_out)

        return output


class FastSurferVINN(FastSurferCNNBase):
    """
    Network Definition of Fully Competitive Network.

    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)

    Attributes
    ----------
    height
        The height of segmentation model (after interpolation layer).
    width
        The width of segmentation model.
    out_tensor_shape
        Out tensor dimensions for interpolation layer.
    interpolation_mode
        Interpolation mode for up/downsampling in flex networks.
    crop_position
        Crop positions for up/downsampling in flex networks.
    inp_block
        Initialized input dense block.
    outp_block
        Initialized output dense block.
    interpol1
        Initialized 2d input interpolation block.
    interpol2
        Initialized 2d output interpolation block.
    classifier
        Initialized Classification Block.

    Methods
    -------
    forward
        Feedforward through graph.
    """

    def __init__(self, params: dict, padded_size: int = 256):
        """
        Construct FastSurferVINN object.

        Parameters
        ----------
        params : Dict
            Dictionary of configurations.
        padded_size : int, default = 256
            Size of image when padded (Default value = 256).
        """
        num_c = params["num_channels"]
        params["num_channels"] = params["num_filters_interpol"]
        super().__init__(params)

        # Flex options
        self.height = params["height"]
        self.width = params["width"]

        self.out_tensor_shape = tuple(
            params.get("out_tensor_" + k, padded_size) for k in ["width", "height"]
        )

        self.interpolation_mode = (
            params["interpolation_mode"]
            if "interpolation_mode" in params
            else "bilinear"
        )
        if self.interpolation_mode not in ["nearest", "bilinear", "bicubic", "area"]:
            raise ValueError("Invalid interpolation mode")

        self.crop_position = (
            params["crop_position"] if "crop_position" in params else "top_left"
        )
        if self.crop_position not in [
            "center",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]:
            raise ValueError("Invalid crop position")

        # Reset input channels to original number (overwritten in super call)
        params["num_channels"] = num_c

        self.inp_block = sm.InputDenseBlock(params)

        params["num_channels"] = params["num_filters"] + params["num_filters_interpol"]
        self.outp_block = sm.OutputDenseBlock(params)

        self.interpol1 = il.Zoom2d(
            (self.width, self.height),
            interpolation_mode=self.interpolation_mode,
            crop_position=self.crop_position,
        )

        self.interpol2 = il.Zoom2d(
            self.out_tensor_shape,
            interpolation_mode=self.interpolation_mode,
            crop_position=self.crop_position,
        )

        # Classifier logits options
        #params["num_channels"] = params["num_filters"]
        params["num_channels"] = params["num_filters"]
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #for mae
        self.enc = MAE_CNN()

        # connector_1 to upsample the skip connection to 2x its resolution
        self.connector_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        # connector_2 to upsample the skip connection to 4x its resolution
        self.connector_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=4,
                               padding=0, bias=True),
            nn.ReLU(True)
        )

        # config_dec1: config for the first decoder after encoding
        config_dec1 = {
            'kernel_h': 3,
            'kernel_w': 3,
            'num_channels': 512,
            'num_filters': 512,
            'num_filters_last': 256,
            'stride_conv': 1,
        }
        # config_dec2, config for the second decoder (final one)
        config_dec2 = {
            'kernel_h': 3,
            'kernel_w': 3,
            'num_channels': 256,
            'num_filters': 256,
            'num_filters_last': 128,
            'stride_conv': 1,
        }
        self.proj_to_classifier = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.Conv2d(128, 71, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )


        
        # upsample the bottleneck feat by 2
        self.upsample_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        # upsample the output of first decoder by 2
        self.upsample_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        self.dec1 = sm.CompetitiveDenseBlock(config_dec1)
        self.dec2 = sm.CompetitiveDenseBlock(config_dec2)
        
        # self.use_softmax = softmax
        # if self.use_softmax:
        #     self.softmax = nn.Softmax(dim=1)

    def forward(
        self, x: Tensor, scale_factor: Tensor, scale_factor_out: Tensor | None = None
    ) -> Tensor:
        """
        Feedforward through graph.

        Parameters
        ----------
        x : Tensor
            Input image [N, C, H, W].
        scale_factor : Tensor
            Tensor of shape [N, 1] representing the scale factor for each image in the
            batch.
        scale_factor_out : Tensor, optional
            Tensor representing the scale factor for the output. Defaults to None.

        Returns
        -------
        logits : Tensor
            Prediction logits.
        """
        # Input block + Flex to 1 mm
        x_resampled, rescale_factor = self.interpol1(x, scale_factor)

        _skip, feat = self.enc(x_resampled)

        feat = self.upsample_1(feat)

        feat = self.dec1(torch.concat([self.connector_1(_skip), feat], dim=1))

        feat = self.upsample_2(feat)

        feat = self.dec2(torch.concat([self.connector_2(_skip), feat], dim=1))
        
        feat =  self.proj_to_classifier(feat)

        if scale_factor_out is None:
            scale_factor_out = rescale_factor
        logits, _ = self.interpol2(feat, scale_factor_out, rescale=True)

        return self.classifier(logits)
    
    def initialize_encoder_from_pretrained(self, pretrained_path):
        print(f"Loading encoder weights from {pretrained_path}")
    
        # Load state dict directly (not wrapped in "model")
        state_dict = torch.load(pretrained_path, map_location="cpu")
    
        # Get the target encoder state_dict (self.enc is MAE_CNN)
        target_dict = self.enc.state_dict()

        # Filter only matching keys
        matched_dict = {
            k: v for k, v in state_dict.items()
            if k in target_dict and v.shape == target_dict[k].shape
        }

        print(f"Matched {len(matched_dict)} / {len(target_dict)} MAE encoder weights")

        # Update and load
        target_dict.update(matched_dict)
        self.enc.load_state_dict(target_dict)

        print("✅ MAE encoder weights loaded into FastSurfer.")


_MODELS = {
    "FastSurferCNN": FastSurferCNN,
    "FastSurferVINN": FastSurferVINN,
}


def build_model(cfg: 'yacs.config.CfgNode') -> FastSurferCNN | FastSurferVINN:
    """
    Build requested model.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Node of configs to be used.

    Returns
    -------
    model
        Object of the initialized model.
    """
    assert (
        cfg.MODEL.MODEL_NAME in _MODELS.keys()
    ), f"Model {cfg.MODEL.MODEL_NAME} not supported"
    params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
    model = _MODELS[cfg.MODEL.MODEL_NAME](params, padded_size=cfg.DATA.PADDED_SIZE)
    return model
