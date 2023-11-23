import init

import sys

sys.path.append("../../")

from encoder.resnet.resnet_ae import ResnetAutoencoder

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from torchsummary import summary

from encoder.dataset import RGBDataset
from encoder.resnet.resnet_ae import ResnetAutoencoder
import torchvision

import encoder.resnet.resnet_encoder as enc
import encoder.resnet.resnet_decoder as dec

input_channels = 3
action_dim = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
netF = enc.ResNet(enc.Bottleneck, [3, 4, 23, 3], return_indices=True)
netF.to(device)

test_input = torch.rand(32, 3, 480, 480).to(device)
# state_dict = torch.load(
#    "model/resnet101.pth"
# )  # https://download.pytorch.org/models/resnet101-63fe2227.pth
# state_dict = torch.load('model/resnet50.pth') # https://download.pytorch.org/models/resnet50-0676ba61.pth
# netF.load_state_dict(state_dict)
# summary(netF, (3, 480, 480))

out, indices = netF(test_input)
print("Feature", out.shape)
print(indices.shape)

print("\n\n\n\n")
netD = dec.ResNetDecoder(dec.Bottleneck, [3, 6, 4, 3])
netD.to(device)

rec = netD(out, indices)
print("Reconstruction", rec.shape)

sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netF.to(device)

# input_size = (3, 224, 224)
# test_input = [torch.rand(2, *input_size).type(torch.FloatTensor).to(device=device)]
out, indices = netF(test_input)

print("Feature shape:", out.shape)

netD = dec.ResNet(dec.Bottleneck, [3, 23, 4, 3])
netD.to(device)
rec = netD(out, indices)

print("Reconstrusted image size:", rec.shape)

# summary(netD, [(2048, 1, 1), (64, 56, 56)])
summary(netF, (3, 221, 221))
summary(netD, (2048, 1, 1))
"""
# resnet50 = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], return_indices=True)
# resnet50.to(device)
resnet50 = ResnetAutoencoder(3)
summary(resnet50, (3, 480, 480))

sys.exit()
# encoder = ResnetEncoder(input_channels=input_channels, action_dim=action_dim)
# decoder = ResnetDecoder(action_dim=action_dim, out_channels=input_channels)
ae = ResnetAutoencoder(
    in_channels=input_channels
)  # (input_shape=(480, 480, 3), z_dim=1024, bottleneck_dim=1024).to("cuda")
ae.to("cuda")

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(
    ae.parameters(),
    lr=1e-3,
    weight_decay=1e-8
    #    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-8
)

# DataLoader is used to load the dataset
# for training
dataset = RGBDataset("/home/kutay/repos/SNeRL/nerf_pretrain/data/drawer-open-v2/rgb")
"""
dataset = torchvision.datasets.CelebA(
    "./tmp/",
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
"""
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)

# summary(encoder, (3, 480, 480))
# summary(decoder, (4,))

summary(ae, (input_channels, 480, 480))

# encoder.train()
# decoder.train()
ae.train()


# train
epochs = 200
outputs = []
losses = []
for epoch in range(epochs):
    mean_loss = 0
    batches = 0
    for imgs in loader:
        imgs.to("cuda")
        imgs = imgs.type(torch.cuda.FloatTensor)
        # Reshaping the image to (-1, 784)
        # image = image.reshape(-1, 28 * 28)
        # Output of Autoencoder
        # image.to("cuda")
        # image = image.type(torch.cuda.FloatTensor)
        # print(type(image))
        reconstructed = ae(imgs)

        # Calculating the loss function
        loss = loss_function(reconstructed, imgs)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        imgs.to("cpu")
        imgs = None
        # Storing the losses in a list for plotting
        losses.append(loss)
        mean_loss += loss
        batches += 1

    torch.cuda.empty_cache()

    mean_loss /= batches
    print("Epoch", epoch, "mean loss", mean_loss)
    outputs.append((epochs, imgs, reconstructed))

# encoder.eval()
# decoder.eval()

torch.save(
    {
        "epoch": epoch,
        "model_state_dict": ae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    },
    f"./tmp/ae_epoch{epoch}_ckpt.pt",
)

torch.save(
    {
        "epoch": epoch,
        "model_state_dict": ae.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    },
    f"./tmp/encoder_epoch{epoch}_ckpt.pt",
)

torch.save(ae.encoder.state_dict(), f"./tmp/encoder_epoch{epoch}.pt")


# Defining the Plot Style
plt.style.use("fivethirtyeight")
plt.xlabel("Iterations")
plt.ylabel("Loss")


# Plotting the last 100 values
plt.plot(losses[-100:])

outputs = np.array(outputs)
losses = np.array(losses)

np.save("./tmp/outputs.npy", outputs)
np.save("./tmp/losses.npy", losses)
"""
