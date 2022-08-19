import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

from layers.blocks import *
from layers.adain import *

class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv_relu(128, 64, 3, 1, 1)
        self.conv3 = conv_relu(64, 32, 3, 1, 1)
        self.conv4 = conv_no_activ(32, 1, 3, 1, 1)

    def forward(self, a, z):
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out


class Decoder(nn.Module):
    def __init__(self, anatomy_out_channels, z_length, num_mask_channels):
        super(Decoder, self).__init__()
        """
        """
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.decoder = AdaINDecoder(self.anatomy_out_channels)

    def forward(self, a, z):
        out = self.decoder(a, z)

        return out


class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes + 1 #background as extra class
        
        self.conv1 = conv_bn_relu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        out = F.softmax(out, dim=1)

        return out


class decVAE(nn.Module):
    def __init__(self, num_output_channels, a_length, dim, upsample_method):
        super(decVAE, self).__init__()
        self.z_len = a_length
        self.dim = dim
        self.upsample_method = upsample_method
        self.num_output_channels = num_output_channels

        #bottleneck
        self.fc = nn.Linear(self.z_len, 1024)
        self.norm = nn.BatchNorm1d(1024)
        self.activ = nn.ReLU(inplace=True)
        #upsample decoder
        self.upsample4 = Interpolate((self.dim//8, self.dim//8), mode=self.upsample_method)
        self.block4 = conv_bn_relu(256, 64, 3, 1, 1)
        self.upsample3 = Interpolate((self.dim//4, self.dim//4), mode=self.upsample_method)
        self.block3 = conv_bn_relu(64, 32, 3, 1, 1)
        self.upsample2 = Interpolate((self.dim//2, self.dim//2), mode=self.upsample_method)
        self.block2 = conv_bn_relu(32, 32, 3, 1, 1)
        self.upsample1 = Interpolate((self.dim, self.dim), mode=self.upsample_method)
        self.block1 = conv_no_activ(32, self.num_output_channels, 3, 1, 1)
    
    def forward(self, z):
        x = self.fc(z)
        #upsample decoder
        x = self.upsample4(x.view(-1, 256, 2, 2))
        x = self.block4(x)
        x = self.upsample3(x)
        x = self.block3(x)
        x = self.upsample2(x)
        x = self.block2(x)
        x = self.upsample1(x)
        x = self.block1(x)
        out = F.gumbel_softmax(x,hard=True,dim=1)

        return x


class AEncoder(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.width = width 
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(self.width, self.height, self.ndf, self.num_output_channels, self.norm, self.upsample)

    def forward(self, x):
        out = self.unet(x)
        out = F.gumbel_softmax(out,hard=True,dim=1)

        return out 


class a_VAE_Encoder(nn.Module):
    def __init__(self, z_length, input_channels=9):
        super(a_VAE_Encoder, self).__init__()
        self.z_length = z_length
        self.input_channels = input_channels

        self.block1 = conv_bn_lrelu(self.input_channels, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(25088, 32)
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


class MEncoder(nn.Module):
    def __init__(self, z_length, input_channels=9):
        super(MEncoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length
        self.input_channels = input_channels

        self.block1 = conv_bn_lrelu(self.input_channels, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(25088, 32)
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

    def forward(self, a, x):
        out = torch.cat([a, x], 1)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class UNet(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, normalization, upsample):
        super(UNet, self).__init__()
        """
        Build the UNet autoencoder
        """
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet(1, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.encoder_block3 = conv_block_unet(self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm)
        self.encoder_block4 = conv_block_unet(self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet(self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm)
        self.decoder_upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet(self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm)
        self.decoder_upsample3 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet(self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm)
        self.decoder_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet(self.ndf * 2, self.ndf, 3, 1, 1, self.norm)
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

    def forward(self, x):
        #encoder
        s1 = self.encoder_block1(x)
        out = self.maxpool(s1)
        s2 = self.encoder_block2(out)
        out = self.maxpool(s2)
        s3 = self.encoder_block3(out)
        out = self.maxpool(s3)
        s4 = self.encoder_block4(out)
        out = self.maxpool(s4)

        #bottleneck
        out = self.bottleneck(out)

        #decoder
        out = self.decoder_upsample1(out)
        out = self.decoder_upconv1(out)
        out = torch.cat((out, s4), 1)
        out = self.decoder_block1(out)
        out = self.decoder_upsample2(out)
        out = self.decoder_upconv2(out)
        out = torch.cat((out, s3), 1)
        out = self.decoder_block2(out)
        out = self.decoder_upsample3(out)
        out = self.decoder_upconv3(out)
        out = torch.cat((out, s2), 1)
        out = self.decoder_block3(out)
        out = self.decoder_upsample4(out)
        out = self.decoder_upconv4(out)
        out = torch.cat((out, s1), 1)
        out = self.decoder_block4(out)
        out = self.classifier_conv(out)

        return out


class SDNet3(nn.Module):
    def __init__(self, width, height, num_classes, ndf, z_length, norm, upsample, anatomy_out_channels, num_mask_channels):
        super(SDNet3, self).__init__()
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

        self.m_encoder = MEncoder(self.z_length)
        self.a_vae_encoder = a_VAE_Encoder(self.z_length, input_channels=self.anatomy_out_channels)
        self.a_vae_decoder = decVAE(self.anatomy_out_channels, self.z_length, self.h, self.upsample)
        self.a_encoder = AEncoder(self.h, self.w, self.ndf, self.anatomy_out_channels, self.norm, self.upsample)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Decoder(self.anatomy_out_channels, self.z_length, self.num_mask_channels)

    def forward(self, x, mask, script_type):
        a_out_spatial = self.a_encoder(x)
        a_vae_out, a_mu_out, a_logvar_out = self.a_vae_encoder(a_out_spatial)
        a_out = self.a_vae_decoder(a_vae_out) 
        seg_pred = self.segmentor(a_out)
        z_out, mu_out, logvar_out = self.m_encoder(a_out, x)
        if script_type == 'training':
            reco = self.decoder(a_out_spatial, z_out)
            a_out_spatial_tilde = self.a_encoder(reco)
            _, mu_out_tilde, _ = self.m_encoder(a_out_spatial_tilde, reco)
        else:
            reco = self.decoder(a_out_spatial, mu_out)
            mu_out_tilde = mu_out #dummy assignment, not needed during validation

        return reco, z_out, mu_out_tilde, a_out_spatial_tilde, a_out, seg_pred, mu_out, logvar_out, a_mu_out, a_logvar_out