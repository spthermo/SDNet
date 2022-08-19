import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

from layers.blocks import *
from layers.adain import *

class AdaINDecoder(nn.Module):
    def __init__(self, h, w, ndf, upsample, anatomy_out_channels, z_length):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.ndf = ndf
        self.h = h
        self.w = w
        self.upsample = upsample

        self.linear = nn.Linear(self.anatomy_out_channels, self.ndf * 4 * 14 * 14)
        self.upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.conv1 = conv_relu(self.ndf * 4, self.ndf * 2, 3, 1, 1)
        self.upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.conv2 = conv_relu(self.ndf * 2, self.ndf, 3, 1, 1)
        self.upsample3 = Interpolate((self.h // 2, self.w  // 2), mode=self.upsample)
        self.conv3 = conv_relu(self.ndf, self.ndf // 2, 3, 1, 1)
        self.upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.conv4 = conv_no_activ(self.ndf // 2, 1, 3, 1, 1)

    def forward(self, a, z):
        out = adaptive_instance_normalization2(a, z)
        out = self.linear(out)
        out = adaptive_instance_normalization2(out, z)
        out = self.upsample1(out.view(-1, self.ndf * 4, 14, 14))
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.upsample2(out)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.upsample3(out)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = self.upsample4(out)
        out = F.tanh(self.conv4(out))

        return out

class Decoder(nn.Module):
    def __init__(self, h, w, ndf, upsample, anatomy_out_channels, z_length):
        super(Decoder, self).__init__()
        self.ndf = ndf
        self.h = h
        self.w = w
        self.upsample = upsample
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.decoder = AdaINDecoder(self.h, self.w, self.ndf, self.upsample, self.anatomy_out_channels, self.z_length)

    def forward(self, a, z):
        out = self.decoder(a, z)

        return out


class Segmentor(nn.Module):
    def __init__(self, h, w, ndf, upsample, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        """
        """
        self.ndf = ndf
        self.h = h
        self.w = w
        self.upsample = upsample
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes + 1 #background as extra class

        self.linear = nn.Linear(self.num_output_channels, self.ndf * 4 * 14 * 14)
        self.seg_upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.seg_block1 = conv_bn_relu(self.ndf * 4, self.ndf * 2, 3, 1, 1)
        self.seg_block1_1 = conv_bn_relu(self.ndf * 2, self.ndf * 2, 3, 1, 1)
        self.seg_upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.seg_block2 = conv_bn_relu(self.ndf * 2, self.ndf, 3, 1, 1)
        self.seg_block2_2 = conv_bn_relu(self.ndf, self.ndf, 3, 1, 1)
        self.seg_upsample3 = Interpolate((self.h // 2, self.w  // 2), mode=self.upsample)
        self.seg_block3 = conv_bn_relu(self.ndf, self.ndf // 2, 3, 1, 1)
        self.seg_block3_3 = conv_bn_relu(self.ndf // 2, self.ndf // 2, 3, 1, 1)
        self.seg_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.seg_block4 = conv_no_activ(self.ndf // 2, self.num_classes, 1, 1, 0)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.seg_upsample1(out.view(-1, self.ndf * 4, 14, 14))
        out = self.seg_block1(out)
        out = self.seg_block1_1(out)
        out = self.seg_upsample2(out)
        out = self.seg_block2(out)
        out = self.seg_block2_2(out)
        out = self.seg_upsample3(out)
        out = self.seg_block3(out)
        out = self.seg_block3_3(out)
        out = self.seg_upsample4(out)
        out = self.seg_block4(out)
        out = F.softmax(out, dim=1)

        return out


class A_Encoder(nn.Module):
    def __init__(self, z_length, ndf):
        super(A_Encoder, self).__init__()
        """
        VAE encoder for the anatomy factors of the image
        z_length: number of spatial (anatomy) factors to encode
        """
        self.z_length = z_length
        self.ndf = ndf

        self.block1 = conv_bn_lrelu(1, self.ndf // 2, 3, 2, 1)
        self.block1_1 = conv_bn_lrelu(self.ndf // 2, self.ndf // 2, 3, 1, 1)
        self.block2 = conv_bn_lrelu(self.ndf // 2, self.ndf, 3, 2, 1)
        self.block2_1 = conv_bn_lrelu(self.ndf, self.ndf, 3, 1, 1)
        self.block3 = conv_bn_lrelu(self.ndf, self.ndf * 2, 3, 2, 1)
        self.block3_1 = conv_bn_lrelu(self.ndf * 2, self.ndf * 2, 3, 1, 1)
        self.block3_2 = conv_bn_lrelu(self.ndf * 2, self.ndf * 2, 3, 1, 1)
        self.block4 = conv_bn_lrelu(self.ndf * 2, self.ndf * 4, 3, 2, 1)
        self.block4_1 = conv_bn_lrelu(self.ndf * 4, self.ndf * 4, 3, 1, 1)
        self.block4_2 = conv_bn_lrelu(self.ndf * 4, self.ndf * 4, 3, 1, 1)
        self.block5 = conv_bn_lrelu(self.ndf * 4, self.ndf * 8, 3, 2, 1)
        self.block5_1 = conv_bn_lrelu(self.ndf * 8, self.ndf * 8, 3, 1, 1)
        self.block5_2 = conv_bn_lrelu(self.ndf * 8, self.ndf * 8, 3, 1, 1)
        self.fc = nn.Linear(self.ndf * 8 * 7 * 7, 32)
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, x):
        out = self.block1(x)
        out = self.block1_1(out)
        out = self.block2(out)
        out = self.block2_1(out)
        out = self.block3(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4(out)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block5(out)
        out = self.block5_1(out)
        out = self.block5_2(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

class M_Encoder(nn.Module):
    def __init__(self, z_length, ndf):
        super(M_Encoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length
        self.ndf = ndf

        self.block1 = conv_bn_lrelu(1, self.ndf // 2, 3, 2, 1)
        self.block2 = conv_bn_lrelu(self.ndf // 2, self.ndf, 3, 2, 1)
        self.block3 = conv_bn_lrelu(self.ndf, self.ndf * 2, 3, 2, 1)
        self.block4 = conv_bn_lrelu(self.ndf * 2, self.ndf * 4, 3, 2, 1)
        self.fc = nn.Linear(self.ndf * 4 * 14 * 14, 32)
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class SDNet2(nn.Module):
    def __init__(self, width, height, num_classes, ndf, z_length, norm, upsample, anatomy_out_channels, num_mask_channels):
        super(SDNet2, self).__init__()
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            num_classes: number of semantice segmentation classes
            z_length: number of modality factors
            anatomy_out_channels: number of anatomy factors
            norm: feature normalization method (BatchNorm)
            ndf: number of feature channels
        """
        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels

        self.m_encoder = M_Encoder(self.z_length, self.ndf // 2)
        self.a_encoder = A_Encoder(self.anatomy_out_channels, self.ndf)
        self.segmentor = Segmentor(self.h, self.w, self.ndf, self.upsample, self.anatomy_out_channels, self.num_classes)
        self.decoder = Decoder(self.h, self.w, self.ndf, self.upsample, self.anatomy_out_channels, self.z_length)

    def forward(self, x, mask, script_type):
        a_out, a_mu_out, a_logvar_out = self.a_encoder(x)
        seg_pred = self.segmentor(a_out)
        z_out, mu_out, logvar_out = self.m_encoder(x)
        if script_type == 'training':
            reco = self.decoder(a_out, z_out)
            _, mu_out_tilde, _ = self.m_encoder(reco)
            _, a_mu_out_tilde, _ = self.a_encoder(reco)
        else:
            reco = self.decoder(a_mu_out, mu_out)
            mu_out_tilde = mu_out #dummy assignment, not needed during validation
            a_mu_out_tilde = a_mu_out #dummy assignment, not needed during validation

        return reco, z_out, mu_out_tilde, a_mu_out_tilde, a_out, seg_pred, mu_out, logvar_out, a_mu_out, a_logvar_out