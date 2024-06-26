"""
Module containing the decoders.
"""
import numpy as np
import torch
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


class DecoderSeq(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 1028) or others.

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(DecoderSeq, self).__init__()

        # Layer parameters
        hid_channels = 128
        downsample_factor = 2
        kernel_size = 4
        hidden_dim = 256
        self.conv_in = hid_channels // (2*downsample_factor)
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[-1]
        L_in = self.img_size[0]
        self.img_size = img_size

        # Convolutional specs
        cnn_kwargs = dict(stride=2, padding=1, dilation=1)
        def conv_output_size(L_in, kernel_size, stride=1, padding=0, dilation=1):
            return (L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride + 1
        L_out1 = conv_output_size(L_in, kernel_size, **cnn_kwargs)
        L_out2 = conv_output_size(L_out1, kernel_size, **cnn_kwargs)
        L_out3 = conv_output_size(L_out2, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, self.conv_in*L_out3)
       
        self.convT1 = nn.ConvTranspose1d(self.conv_in, hid_channels//downsample_factor, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose1d(hid_channels//downsample_factor, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose1d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, self.conv_in, -1) # NOTE: don't reshape unpack as in 2D convolution, 16 channels required

        # Convolutional layers with ReLu activations
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))
        # x = self.convT3(x) # NOTE: for stability use logits
        x = x.permute(0, 2, 1) # assert [N, L, D]
        return x


class DecoderSeqpooled(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model

        Parameters
        ----------
        img_size : tuple of ints
            Size of pooled embedded sequences. E.g. (1, D) or others.

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(DecoderSeqpooled, self).__init__()

        # Layer parameters
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        n_chan = self.img_size[-1]
        L_in = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, L_in)


    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        x = x.reshape(batch_size, self.img_size[0], 1)

        return x


