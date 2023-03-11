import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import os
import sys
import logging

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    running_samples=0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            running_samples+=len(inputs)
            if running_samples>(0.1*len(test_loader.dataset)):
                break

    total_loss = running_loss / running_samples
    total_acc = running_corrects / running_samples
    logger.info(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 128  == 0:
                    accuracy = running_corrects/running_samples
                    logger.info("Epoch: {} Step: {} Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            epoch,
                            step,
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.1*len(image_dataset[phase].dataset)):
                    break
                    
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
        if loss_counter==1:
            break
    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False 

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 6))
    return model


def create_data_loaders(data_dir, batch_size, transform):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return dataloader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    logger.info(f"Hyperparameters: Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}")
    model=net()
    model=model.to(device)

    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
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
    logger.info("Start Training.")
    model=train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Get test data loader")
    test_loader = create_data_loaders(args.data_dir+'/test', args.batch_size, testing_transform)
    test(model, test_loader, criterion, device)

    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    print("model dir:", os.path.join(model_dir, "model.pth"))
    model = models.resnet50()
    for param in model.parameters():
        param.requires_grad = False 

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 6))
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 5)",
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
