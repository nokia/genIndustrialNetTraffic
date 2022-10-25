import torch
import torch.nn as nn
import copy

## (Conditional) Variational Autoencoder class and encoder/decoder class

class VAE(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim, encoder_layersizes, decoder_layersizes, normparams, conditional=False):
        super().__init__()

        if conditional:
            assert y_dim > 0

        assert type(latent_dim) == int
        assert type(x_dim) == int

        self.latent_dim = latent_dim

        self.encoder = Encoder(x_dim, y_dim, latent_dim, conditional, encoder_layersizes, normparams) 
        self.decoder = Decoder(x_dim, y_dim, latent_dim, conditional, decoder_layersizes, normparams)
        self.clamp = [-4, 15]

    def inference(self, z, y=None):
        recon_x = self.decoder(z, y)
        return recon_x

class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim, conditional, encoder_layersizes, normparams):
        """
        Args:
            x_dim (int): e.g., x variable (to be reconstructed) dimension
            y_dim (int): e.g., y variable (if x is conditioned by y) dimension
            latent_dim (int): latent dimension
            conditional (bool): whether to use conditional VAE, if conditional then x is conditioned by y
        """
        super().__init__()

        self.conditional = conditional
        self.normparams = normparams

        layer_sizes = copy.deepcopy(encoder_layersizes)
        layer_sizes.insert(0, x_dim)
        if self.conditional:
            layer_sizes[0] += y_dim

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.liner_means = nn.Linear(layer_sizes[-1], latent_dim)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim, conditional, decoder_layersizes, normparams):
        """
        Args:
            x_dim (int): e.g., x variable (to be reconstructed) dimension
            y_dim (int): e.g., y variable (if x is conditioned by y) dimension
            latent_dim (int): latent dimension
            conditional (bool): whether to use conditional VAE, if conditional then x is conditioned by y
        """
        super().__init__()

        self.conditional = conditional
        self.normparams = normparams

        layer_sizes = copy.deepcopy(decoder_layersizes)
        layer_sizes.insert(0, latent_dim)
        if self.conditional:
            layer_sizes[0] += y_dim
        
        layer_sizes.append(x_dim)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes[:-1]):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, y=None):
        if self.conditional:
            y = idx2onehot(y, 5)
            z = torch.cat((z, y), dim=-1)
        x = self.MLP(z)
        return x

## GAN class definition and corresponding Generator and Discriminator

class GAN(nn.Module):
     def __init__(self, sample_dim, latent_dim, generator_layersizes, discriminator_layersizes):
        super().__init__()

        assert type(sample_dim) == int
        assert type(latent_dim) == int
        self.sample_dim = sample_dim
        self.latent_dim = latent_dim
        self.generator_layersizes = generator_layersizes
        self.discriminator_layersizes = discriminator_layersizes

     def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

     def initialize_generator(self, device):
        generator = Generator(self.latent_dim, self.sample_dim, self.generator_layersizes)
        generator.apply(self.weights_init)
        generator.to(device)
        return generator    

     def generate_samples(self, generator, num_samples, latent_dim, device):
        noise = torch.randn(num_samples, latent_dim, device=device)
        generator.eval()
        samples = generator(noise)
        samples = samples.detach().numpy()
        return samples

class Generator(nn.Module):
  """
    DCGan Generator
  """
  def __init__(self, input_dim, out_dim, generator_layersizes):
    super().__init__()

    layer_sizes = copy.deepcopy(generator_layersizes)
    layer_sizes.insert(0, input_dim)
    layer_sizes.append(out_dim)

    self.MLP = nn.Sequential()

    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        self.MLP.add_module(
            name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        if i + 1 < len(layer_sizes[:-1]):
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())
        else:
            self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())


  def forward(self, x):
    """Forward pass"""
    return self.MLP(x)
    # return self.layers(x)


class Discriminator(nn.Module):
  """
    DCGan Discriminator
  """
  def __init__(self, input_dim, out_dim, discriminator_layersizes):
    super().__init__()

    assert(out_dim==1)
    layer_sizes = copy.deepcopy(discriminator_layersizes)
    layer_sizes.insert(0, input_dim)
    layer_sizes.append(out_dim)

    self.MLP = nn.Sequential()

    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        self.MLP.add_module(
            name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        if i + 1 < len(layer_sizes[:-1]):
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())
            self.MLP.add_module(name="Dropout{:d}".format(i), module=nn.Dropout(0.3))
        else:
            self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())


  def forward(self, x):
    """Forward pass"""
    return self.MLP(x)
    # return self.layers(x)


## Additional functions 

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot
