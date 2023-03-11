'''Script for training image classifier'''
import argparse
import io
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import smdebug.pytorch as smd
except:
    print("can't import smdebug")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device, hook):
    model.eval()
    if hook:
        hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer, epoch, device, hook):
    model.train()
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data=data.to(device)
        target=target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )


def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False 
    model.fc.requires_grad = True
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 6))
    return model


def create_data_loaders(data_dir, batch_size, transform, sampler=None):
    shuffle = True if sampler is None else False
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, sampler=sampler)
    return dataloader

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    model=net()
    model=model.to(device)

    criterion = nn.CrossEntropyLoss()
    try:
        import smdebug
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
        hook.register_loss(criterion)
    except:
        hook = None

    training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    logger.info("Get train data loader")
    train_loader = create_data_loaders(args.data_dir+'/train', args.batch_size, training_transform)
    logger.info("Get valid data loader")
    valid_loader = create_data_loaders(args.data_dir+'/val', args.batch_size, testing_transform)
    logger.info("Get test data loader")
    test_loader = create_data_loaders(args.data_dir+'/test', args.batch_size, testing_transform)

    train_dataset = ImageFolder(args.data_dir+'/train', transform=training_transform)

    # Create a weighted sampler to handle class imbalance
    class_counts = torch.bincount(torch.tensor(train_dataset.targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = create_data_loaders(args.data_dir+'/train', args.batch_size, training_transform, sampler=sampler)

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, device, hook)
        test(model, test_loader, criterion, device, hook)
        # Update the learning rate scheduler
        scheduler.step()

    # Save the trained model
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    print("model dir:", os.path.join(model_dir, "model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    for param in model.parameters():
        param.requires_grad = False

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 6))
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def load_from_bytearray(request_body):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


def input_fn(request_body, request_content_type):
    # if set content_type as "image/jpg" or "application/x-npy",
    # the input is also a python bytearray
    if request_content_type == "application/x-image":
        image_tensor = load_from_bytearray(request_body)
    else:
        print("not support this type yet")
        raise ValueError("not support this type yet")
    return image_tensor


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    model.eval()
    with torch.no_grad():
        output = model.forward(input_object)
    pred = output.max(1, keepdim=True)[1]

    return {"predictions": pred.item()}


# Serialize the prediction result into the desired response content type
def output_fn(predictions, response_content_type):
    return json.dumps(predictions)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args=parser.parse_args()
    main(args)

