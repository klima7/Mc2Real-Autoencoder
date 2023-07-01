import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        
        ch = base_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, ch, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            act_fn(),
            
            nn.Conv2d(ch, ch*2, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(ch*2, ch*2, kernel_size=3, padding=1),
            act_fn(),
            
            nn.Conv2d(ch*2, ch*4, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, padding=1),
            act_fn(),
            
            nn.Conv2d(ch*4, ch*8, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Conv2d(ch*8, ch*8, kernel_size=3, padding=1),
            act_fn(),
            
            nn.Flatten(),
            nn.Linear(16 * 8 * ch, 2*latent_dim),
            act_fn(),
            nn.Linear(2*latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class Decoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        ch = base_channels
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            act_fn(),
            nn.Linear(2*latent_dim, 16 * 8 * ch),
            act_fn(),
        )
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*8, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(ch*8, ch*4, kernel_size=3, padding=1),
            act_fn(),
            
            nn.ConvTranspose2d(ch*4, ch*4, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(ch*4, ch*2, kernel_size=3, padding=1),
            act_fn(),
            
            nn.ConvTranspose2d(ch*2, ch*2, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(ch*2, ch, kernel_size=3, padding=1),
            act_fn(),
            
            nn.ConvTranspose2d(ch, ch, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
            act_fn(),
            nn.Conv2d(ch, input_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    

class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        num_input_channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim)
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, 64, 64)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def predict(self, np_img):
        """
        np_img: numpy image of shape (64, 64, 3), range 0-1
        return: image, same as input
        """
        np_img = np.transpose(np_img - 0.5, [2, 0, 1])
        img = self(torch.from_numpy(np_img).unsqueeze(0).to(self.device)).squeeze().detach().cpu().numpy()
        img = np.transpose(img, [1, 2, 0]) + 0.5
        return img

    def _get_reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
