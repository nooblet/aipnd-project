from time import time
import argparse
import logging
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import models, transforms, datasets
from collections import OrderedDict
from PIL import Image
import cattoname as ctn

#Create logger
logger = logging.getLogger('train_app')
logger.setLevel(logging.DEBUG)
#Console stream handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch.setFormatter(formatter)

logger.addHandler(ch)

#models
vgg16 = models.vgg16(pretrained=True)
#print(vgg16)
densenet161 = models.densenet161(pretrained=True)
#print(densenet161)
brain = {'vgg16':vgg16, 'densenet161':densenet161}

def main():
    logger.info('Starting training application')
    #Define start_time to measure total runtime
    start_time = time()

    #Get parse command line arguements
    in_arg = get_input_args()
    logger.debug(in_arg)

    logger.info('Preparing data sets')
    dataloaders, class_to_idx = prepare_data(in_arg.data_directory)

    #Prepare the Network
    logger.info('Preparing Neural network')
    chkp, loaded = load_checkpoint(in_arg.arch, in_arg.save_dir)
    if (chkp is None) and (loaded is False):
        prepare_network(in_arg.arch, in_arg.hidden_units)
    brain[in_arg.arch].class_to_idx = class_to_idx

    #Begin network training
    logger.info('Begin network training, GPU mode: {}'.format(in_arg.gpu))
    logger.warn('GPU detected: {}'.format(torch.cuda.is_available()))
    begin_learning(in_arg.arch, dataloaders, in_arg.learning_rate, in_arg.epochs, in_arg.gpu, in_arg.save_dir, chkp)

    logger.info("Total application runtime: {:.3f}".format(time()-start_time))

def get_input_args():
    """
    Function for parsing command line arguments
    Command line arguments to be accepted:
    Required
    data_directory - Directory path of training images

    Optional
    save_dir - Directory to save checkpoint
    arch - CNN model architecture to use
    learning_rate - Speed of learning
    hidden_units - Number of hidden nodes
    epochs - Number of times training will be run
    gpu - Bool argument to indicate usage of GPU

    Parameters:
    None
    Returns:
    parse_args() - data structure that stores the command line arguments object
    """
    #Create an argument parser
    parser = argparse.ArgumentParser(description='Options that modifies the training')
    #arg1: data_directory
    parser.add_argument(
        'data_directory',
        type=str,
        help='Provide directory path to training data'
    )

    #arg2: save_dir
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoint.pth',
        help='Provide directory path to save checkpoint'
    )

    #arg3: arch
    parser.add_argument(
        '--arch',
        type=str,
        default='vgg16',
        help='CNN Architecture to use: densenet161, vgg16'
    )

    #arg4: learning_rate
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Provide a floating point number for the learning rate'
    )

    #arg5: hidden_units
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=1024,
        help='Number of hidden units in Nueral network'
    )

    #arg6: epochs
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of train run'
    )

    #arg7: gpu
    parser.add_argument(
        '--gpu',
        type=bool,
        default=True,
        help='Boolean value to select if GPU should be used'
    )
    return parser.parse_args()

def prepare_data(data_dir):
    """
    Prepares the data to be used for training

    Parameter:
    data_dir - location of the images to be used for training
    Returns:
    dataloaders - data to be used for training
    class_to_idx - mapping of class to idx
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose(
        [transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    )
    data_transforms = transforms.Compose(
        [transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])]
    )
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=train_transforms),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms)
    }

    dataloaders = {
        'train_loader':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid_loader':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    }

    return dataloaders,image_datasets['train'].class_to_idx

def prepare_network(arch, hidden_units):
    """
    Prepare network, classifier
    """
    if arch == 'vgg16':
        input_size = 25088
    elif arch == 'densenet161':
        input_size = 2208
    else:
        logger.error('Invalid model architecture')
        exit()
    num_of_flowers = 102
    for param in brain[arch].parameters():
        param.requires_grad = False
    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(in_features=input_size, out_features=hidden_units)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(in_features=hidden_units, out_features=hidden_units)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(0.01)),
            ('fc3', nn.Linear(in_features=hidden_units, out_features=num_of_flowers)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )
    brain[arch].classifier = classifier

def begin_learning(arch, dataloader, learnrate, epoch, gpu, save_dir, chkp):
    """
    Function training the neural network
    """
    best_accuracy = 0
    cuda = torch.cuda.is_available()
    logger.info('Switching neural model to train mode')
    brain[arch].train()
    print_time = 10
    if cuda is True:
        logger.info('Running on GPU mode')
        brain[arch].cuda()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(brain[arch].classifier.parameters(), learnrate)
        if chkp is not None:
            optimizer.load_state_dict(chkp['optimizer_state'])
        training_start = time()
        for step in range(epoch):
            running_loss = 0
            for itr, (inputs, labels) in enumerate(dataloader['train_loader']):
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                #Clear the gradients from all Variables
                optimizer.zero_grad()
                #Make a forward pass
                output = brain[arch].forward(inputs)
                #Calculate loss
                loss = criterion(output, labels)
                #Perform back propagation
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

                if (itr+1)%print_time == 0:
                    logger.info('Switching neural model to eval mode')
                    brain[arch].eval()
                    accuracy = 0
                    validation_loss = 0
                    for v_input, v_label in iter(dataloader['valid_loader']):
                        v_input, v_label = Variable(v_input, volatile=True), Variable(v_label, volatile=True)
                        v_input, v_label = v_input.cuda(), v_label.cuda()

                        output = brain[arch].forward(v_input)
                        validation_loss += criterion(output, v_label).data[0]

                        ps = torch.exp(output).data

                        equality = (v_label.data == ps.max(1)[1])

                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        print("Epoch: {}/{}...".format(step+1, epoch),
                        "Train loss: {:.3f}".format(running_loss/print_time),
                        "Validation loss: {:.3f}".format(validation_loss/len(dataloader['valid_loader'])),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(dataloader['valid_loader'])))
                    running_loss = 0
                    best_accuracy = accuracy/len(dataloader['valid_loader'])
                    save_checkpoint(arch, best_accuracy, optimizer, save_dir)
                    logger.info('Switching neural model back to train mode')
                    brain[arch].train()
        logger.info('Training time: {}'.format(time()-training_start))
    else:
        logger.info('Running on CPU mode')
        brain[arch].cpu()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(brain[arch].classifier.parameters(), learnrate)
        if chkp is not None:
            optimizer.state_dict = chkp['optimizer_state']
        training_start = time()
        for step in range(epoch):
            running_loss = 0
            for itr, (inputs, labels) in enumerate(dataloader['train_loader']):
                inputs, labels = Variable(inputs), Variable(labels)

                #Clear the gradients from all Variables
                optimizer.zero_grad()
                #Make a forward pass
                output = brain[arch].forward(inputs)
                #Calculate loss
                loss = criterion(output, labels)
                #Perform back propagation
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

                if (itr+1)%print_time == 0:
                    logger.info('Switching neural model to eval mode')
                    brain[arch].eval()
                    accuracy = 0
                    validation_loss = 0
                    for v_input, v_label in iter(dataloader['valid_loader']):
                        v_input, v_label = Variable(v_input, volatile=True), Variable(v_label, volatile=True)

                        output = brain[arch].forward(v_input)
                        validation_loss += criterion(output, v_label).data[0]

                        ps = torch.exp(output).data[0]

                        equality = (v_label.data == ps.max(1)[1])

                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        print("Epoch: {}/{}...".format(step+1, epoch),
                        "Train loss: {:.3f}".format(running_loss/print_time),
                        "Validation loss: {:.3f}".format(validation_loss/len(dataloader['valid_loader'])),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(dataloader['valid_loader'])))
                    running_loss = 0
                    best_accuracy = accuracy/len(dataloader['valid_loader'])
                    save_checkpoint(arch, best_accuracy, optimizer, save_dir)
                    logger.info('Switching neural model back to train mode')
                    brain[arch].train()
        logger.info('Training time: {:.3f}'.format(time()-training_start))

def save_checkpoint(arch, best_accuracy, optimizer, save_dir):
    """
    Function for saving checkpoint when accuracy increased
    """
    if brain[arch].best_accuracy < best_accuracy:
        brain[arch].best_accuracy = best_accuracy
        checkpoint = {
            "state_dict":brain[arch].state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "classifier":brain[arch].classifier,
            "best_accuracy":best_accuracy,
            "class_to_idx":brain[arch].class_to_idx
        }
        logger.info('Saving checkpoint')
        torch.save(checkpoint, arch+'_'+save_dir)
    else:
        pass

def load_checkpoint(arch, save_dir):
    """
    Loads saved checkpoint of network model
    """
    try:
        checkpoint = torch.load(arch+'_'+save_dir)
    except Exception as e:
        logger.error('{}:No checkpoint found in the given path: {}'.format(e, arch+'_'+save_dir))
        loaded = False
        brain[arch].best_accuracy = 0
        return None, loaded
    else:
        loaded = True
        brain[arch].classifier = checkpoint['classifier']
        brain[arch].load_state_dict(checkpoint['state_dict'])
        brain[arch].best_accuracy = checkpoint['best_accuracy']
        brain[arch].class_to_idx = checkpoint['class_to_idx']
        return checkpoint, loaded


if __name__ == "__main__":
    main()