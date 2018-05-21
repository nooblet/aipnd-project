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
import json
import numpy as np

#Create logger
logger = logging.getLogger('train_app')
logger.setLevel(logging.DEBUG)
#Console stream handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch.setFormatter(formatter)

logger.addHandler(ch)

def main():
    logger.info('Starting predict application')
    #Define start_time to measure total runtime
    start_time = time()

    #Get parse command line arguements
    in_arg = get_input_args()
    logger.debug(in_arg)
    #Preparation part
    name_dict = get_flower_dict(in_arg.category_names)

    #Prepare the image
    torch_img = process_image(in_arg.input)
    #Load model checkpoint
    model = load_model(in_arg.checkpoint)

    scores, category = predict(model, torch_img, in_arg.top_k, in_arg.gpu)

    #Format the output
    present_result(scores, category, name_dict)
    
    logger.info("Total application runtime: {:.3f}".format(time()-start_time))

def get_input_args():
    """
    Function for parsing command line arguments
    Command line arguments to be accepted:
    Required
    input - image file name
    checkpoint - checkpoint file to reconstruct model

    Optional
    top_k - number of possible values to show
    category_names - file to use for mapping category to name
    gpu - Bool argument to indicate usage of GPU

    Parameters:
    None
    Returns:
    parse_args() - data structure that stores the command line arguments object
    """
    #Create an argument parser
    parser = argparse.ArgumentParser(description='Options that modifies the prediction')
    #arg1: input
    parser.add_argument(
        'input',
        type=str,
        help='Provide image filename'
    )

    #arg2: checkpoint
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Provide checkpoint file: <model>_checkpoint.pth'
    )

    #arg3: top_k
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of possible category candidates'
    )

    #arg4: category_names
    parser.add_argument(
        '--category_names',
        type=str,
        default='cat_to_name.json',
        help='Provide a json file for mapping category to name'
    )

    #arg5: gpu
    parser.add_argument(
        '--gpu',
        type=bool,
        default=True,
        help='Boolean value to select if GPU should be used'
    )
    return parser.parse_args()

def get_flower_dict(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    img = Image.open(image)
    
    img = img.resize((256, 256))
    left = (256-224)/2
    top = (256-224)/2
    right = (256+224)/2
    bottom = (256+224)/2
    img = img.crop((left, top, right, bottom))

    img = np.array(img)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img/255 - mean) / std

    img = img.transpose((2,0,1))

    return torch.from_numpy(img)

def load_model(checkpoint):
    """
    Return:
    model
    """
    load_checkpoint = torch.load(checkpoint)
    arch = checkpoint.split('_')[0]
    if arch == 'vgg16':
        loaded_model = models.vgg16(pretrained=True)
    elif arch == 'densenet161':
        loaded_model = models.densenet161(pretrained=True)
    else:
        logger.error('Unsupported model detected extracted from checkpoint')
        exit()
    loaded_model.classifier = load_checkpoint['classifier']
    loaded_model.load_state_dict(load_checkpoint['state_dict'])
    loaded_model.best_accuracy = load_checkpoint['best_accuracy']
    loaded_model.class_to_idx = load_checkpoint['class_to_idx']

    return loaded_model

def predict(model, img, top_k, gpu):
    tmp_class, tmp_prob = list(), list()

    model.eval()
    img.unsqueeze_(0)
    data = Variable(img, volatile=True)
    data = data.type(torch.FloatTensor)
    if gpu and torch.cuda.is_available() is True:
        model.cuda()
        data = data.cuda()
    else:
        model.cpu()
    
    output = model.forward(data)
    ps = torch.exp(output)
    tensor_prob, tensor_class = ps.topk(top_k)

    inv_class = {v:k for k, v in model.class_to_idx.items()}

    tmp_class = [inv_class[k] for k in tensor_class[0].data]
    for prob in tensor_prob[0].data:
        tmp_prob.append(round(prob,8))

    return tmp_prob, tmp_class

def present_result(scores, category, name_dict):
    index = 0
    for score, cat in zip(scores, category):
        print('{:2d} Name: {:20s} Score: {:.8f}'.format(index+1, name_dict[cat], score))
        index += 1

if __name__ == "__main__":
    main()
