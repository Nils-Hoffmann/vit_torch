import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn.functional as F
from model import ModelConfig, ViT
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def main():
    transform = Compose([Resize((320, 320)), ToTensor()])
    training_data = datasets.Imagenette(
        root="/tmp/data",
        split="train",
        size="320px",
        download=True,
        transform=transform,
    )
    test_data = datasets.Imagenette(
        root="/tmp/data", split="val", size="320px", download=True, transform=transform
    )

    epochs = 8
    logging_interval = 100
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1
    )

    def train_epoch(model, device, train_loader, opt, epoch):

        model.train()
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(data.transpose(1, 3))
                loss = F.cross_entropy(output.float(), target)

            loss.backward()
            opt.step()
            if step % logging_interval == 0:
                print(
                    f"training epoch {epoch}, step {step} / {len(train_loader.dataset)}, loss {loss}"
                )

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output.float(), target)
            test_loss /= len(test_loader.dataset)

    device = "cuda"
    config = ModelConfig(image_res=(320, 320, 3))
    model = ViT(config).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for e in range(epochs):
        train_epoch(model, device, train_loader, optimizer, e)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()
