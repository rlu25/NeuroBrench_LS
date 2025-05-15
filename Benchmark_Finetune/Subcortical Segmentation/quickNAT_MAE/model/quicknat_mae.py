"""Quicknat architecture"""
import numpy as np
import torch
import torch.nn as nn
from . import nn_modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se
from .mae_cnn import MAE_CNN

class QuickNat(nn.Module):
    """
    A PyTorch implementation of QuickNAT

    """
    def __init__(self, params):
        """

        :param params: {'num_channels':1,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_classes':28
                        'se_block': False,
                        'drop_out':0.2}
        """
        super(QuickNat, self).__init__()
        print(se.SELayer(params['se_block']))
        self.encode1 = sm.EncoderBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.EncoderBlock(params, se_block_type=params['se_block'])
        self.encode3 = sm.EncoderBlock(params, se_block_type=params['se_block'])
        self.encode4 = sm.EncoderBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = 512
        self.bottleneck = sm.DenseBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = params['num_filters'] * 2
        self.decode1 = sm.DecoderBlock(params, se_block_type=params['se_block'])
        self.decode2 = sm.DecoderBlock(params, se_block_type=params['se_block'])
        self.decode3 = sm.DecoderBlock(params, se_block_type=params['se_block'])
        self.decode4 = sm.DecoderBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # mae
        self.enc = MAE_CNN()
        # connector_1 to upsample the skip connection to 2x its resolution
        self.connector_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        # connector_2 to upsample the skip connection to 4x its resolution
        self.connector_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 64, kernel_size=4, stride=4,
                               padding=0, bias=True),
            nn.ReLU(True)
        )
        
        # upsample the bottleneck feat by 2
        self.upsample_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        # upsample the output of first decoder by 2
        self.upsample_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
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

        print("✅ MAE encoder weights loaded into QuickNat.")

    def forward(self, input):
        """

        :param input: X
        :return: probabiliy map
        """
        _skip, feat = self.enc(input)

        bn = self.bottleneck.forward(feat)

        d3 = bn

        out2 = self.connector_1(_skip)
        d3 = self.upsample_1(d3)
        d2 = self.decode2.forward(d3, out2, None)

        out1 = self.connector_2(_skip)
        d2 = self.upsample_2(d2)
        d1 = self.decode3.forward(d2, out1, None)
        prob = self.classifier.forward(d1)

        return prob
        

    def enable_test_dropout(self):
        """
        Enables test time drop out for uncertainity
        :return:
        """
        attr_dict = self.__dict__['_modules']
        for i in range(1, 5):
            encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def predict(self, X, device=0, enable_dropout=False, out_prob=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()

        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        if out_prob:
            return out
        else:
            max_val, idx = torch.max(out, 1)
            idx = idx.data.cpu().numpy()
            prediction = np.squeeze(idx)
            del X, out, idx, max_val
            return prediction
