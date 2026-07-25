import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
from pathlib import Path
from multiprocessing import freeze_support
from tqdm import tqdm
from torchvision.utils import save_image
from torch.optim import Adam
import torchvision.transforms as transforms

data_root = Path("data")
checkpoint_path = Path("outputs") / "vae_trained.pt"
recon_path = "vae_test_reconstructions.png"

cuda = True
device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 128
x_dim = 28 * 28
hidden_dim = 256
latent_dim = 96
lr = 1e-3
epochs = 100

mnist_transform = transforms.ToTensor()  # MNIST is already 1-channel 28x28 in [0, 1]

# Gaussian MLP Encoder & Decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_input3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        x = x.view(x.size(0), -1)  # accept (N,C,H,W) or already-flat input
        x = self.LeakyReLU(self.fc_input(x))
        x = self.LeakyReLU(self.fc_input2(x))
        x = self.LeakyReLU(self.fc_input3(x))
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        return mean, var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.LeakyReLU(self.fc_hidden(x))
        h = self.LeakyReLU(self.fc_hidden2(h))
        h = self.LeakyReLU(self.fc_hidden3(h))
        x_hat = self.Sigmoid(self.fc_output(h))
        return x_hat.reshape(-1, 1, 28, 28)


# Convolutional Gaussian Encoder & Decoder (for RGB / larger images, e.g. CelebA).
# Same (mean, log_var) / (N,C,S,S) interface as the MLP pair, so Model, loss, and
# the OT corrector work with either. S must be a power of 2 >= 8.

class ConvEncoder(nn.Module):
    def __init__(self, num_channels, image_size, latent_dim, base=64):
        super(ConvEncoder, self).__init__()
        self.num_channels, self.image_size = num_channels, image_size
        n = int(round(math.log2(image_size // 4)))          # stride-2 stages down to 4x4
        chans = [num_channels] + [base * (2 ** i) for i in range(n)]
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        self.conv = nn.Sequential(*layers)
        feat = chans[-1] * 4 * 4
        self.fc_mean = nn.Linear(feat, latent_dim)
        self.fc_var = nn.Linear(feat, latent_dim)

    def forward(self, x):
        if x.dim() == 2:                                    # accept flattened input too
            x = x.view(-1, self.num_channels, self.image_size, self.image_size)
        h = self.conv(x).flatten(1)
        return self.fc_mean(h), self.fc_var(h)


class ConvDecoder(nn.Module):
    def __init__(self, num_channels, image_size, latent_dim, base=64):
        super(ConvDecoder, self).__init__()
        n = int(round(math.log2(image_size // 4)))
        chans = [base * (2 ** (n - 1 - i)) for i in range(n)]
        self.start_ch = chans[0]
        self.fc = nn.Linear(latent_dim, chans[0] * 4 * 4)
        layers, prev = [], chans[0]
        for i in range(1, n):
            layers += [nn.ConvTranspose2d(prev, chans[i], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            prev = chans[i]
        layers += [nn.ConvTranspose2d(prev, num_channels, 4, 2, 1), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, 4, 4)
        return self.deconv(h)


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(encoder=encoder, decoder=decoder).to(device)

def loss_function(x, x_hat, mean, log_var):
    x = x.view(x.size(0), -1)
    x_hat = x_hat.view(x_hat.size(0), -1)
    # Sum over pixels / latent dims, average over the batch, so BCE and KLD
    # are on the same scale (per-image ELBO).
    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

optimizer = Adam(model.parameters(), lr=lr)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "device": str(next(model.parameters()).device),
    }, path)


def load_checkpoint(path: Path, device: torch.device) -> Model:
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    loaded_model = Model(encoder=encoder, decoder=decoder).to(device)
    checkpoint = torch.load(path, map_location=device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    return loaded_model

def main() -> None:
    kwargs = {'num_workers': 4, 'pin_memory': (device.type == 'cuda')}

    train_dataset = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=mnist_transform,
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=mnist_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs,
    )

    print("Training VAE....")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
            x = x.view(x.size(0), x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / len(train_loader))

    save_checkpoint(model, checkpoint_path)
    print(f"Saved trained model to {checkpoint_path}")

    print("Loading trained model for test evaluation and reconstructions...")
    loaded_model = load_checkpoint(checkpoint_path, device)

    test_loss = 0.0
    reconstruction_batches = []
    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc="Evaluating"):
            x = x.view(x.size(0), x_dim).to(device)
            x_hat, mean, log_var = loaded_model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            test_loss += loss.item() * x.size(0)
            reconstruction_batches.append((x, x_hat))

    test_loss /= len(test_dataset)
    print(f"Test loss: {test_loss:.6f}")

    images, reconstructions = reconstruction_batches[0]
    comparison = torch.cat([
        images[:8].view(-1, 1, 28, 28),
        reconstructions[:8].view(-1, 1, 28, 28),
    ], dim=0)
    save_image(comparison, recon_path, nrow=8)
    print(f"Saved test reconstructions to {recon_path}")
    print("Finished Training!")


if __name__ == "__main__":
    freeze_support()
    main()