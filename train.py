import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

# Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learning_rate = 2e-4
Batch_size = 128
Image_size = 64
Channels_img = 3
Z_dim = 100
Num_epochs = 5
Features_disc = 64
Features_gen = 64

transforms = transforms.Compose(
    [
        transforms.Resize(Image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(Channels_img)], [0.5 for _ in range(Channels_img)]
        ),
    ]
)

# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root="celeb_a", transform=transforms)
loader = DataLoader(dataset, batch_size=Batch_size, shuffle=True)
gen = Generator(Z_dim, Channels_img, Features_gen).to(device)
disc = Discriminator(Channels_img, Features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=Learning_rate, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=Learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(Num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        print(f"Batch index: {batch_idx}")
        real = real.to(device)
        noise = torch.randn((Batch_size, Z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # train discriminator: max log(D(x)) + log(1 - D(G(z))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # train generator: max log(D(G(Z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{Num_epochs}] Batch [{batch_idx}/{len(loader)}]"
                f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # pick up to 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1


