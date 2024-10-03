import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.optim import SGD, Adam, RMSprop, AdamW
import argparse
from tqdm import tqdm
from torch.nn.utils.parametrizations import spectral_norm
import time
import numpy as np
from PIL import Image
import os

def mnist_digit_dataset(digit):
    mnist_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        # torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    mnist_train = torchvision.datasets.MNIST(..., train=True, download=True, transform=mnist_transform)
    mnist_train_target = mnist_train.targets
    classidx = mnist_train_target == digit

    digit_list = []
    for i in range(len(classidx)):
        if classidx[i] == True:
            digit_list.append(i)
    digit_list = torch.tensor(digit_list)

    train_dataset = torch.utils.data.Subset(mnist_train, digit_list)
    return train_dataset

class MLP(nn.Module):
    def __init__(self, num_features):
        super(MLP, self).__init__()

        self.hidden_feature = int(num_features / 100)
        self.linear1 = nn.Linear(num_features, self.hidden_feature)
        self.linear2 = nn.Linear(self.hidden_feature, self.hidden_feature)
        self.linear3 = nn.Linear(self.hidden_feature, self.hidden_feature)
        self.linear4 = nn.Linear(self.hidden_feature, self.hidden_feature)
        self.linear5 = nn.Linear(self.hidden_feature, 1)


        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.act(x)
        x = self.linear4(x)
        x = self.act(x)
        x = self.linear5(x)
        # x = self.act(x)
        return x

class Quad(nn.Module):
    def __init__(self, num_features, lambda2):
        super(Quad, self).__init__()
        self.para = Parameter(torch.zeros((num_features, 1)), requires_grad=True)
        self.lambda2 = lambda2
    
    def forward(self, x):
        output = 0.5 * self.lambda2 * (torch.norm(x, p=2, dim=1) ** 2) + torch.matmul(x, self.para).squeeze()
        return output

def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))    

    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return spectral_norm(m)
    else:
        return m

def delta_l2(net, x, stepsize, LAMBDA2, LAMBDA1=0, normalize=True):
    # from x to y
    delta = torch.zeros(x.shape)
    delta = delta.cuda()
    delta = Parameter(delta, requires_grad = True)
        
    for i in range(100):
        loss_objective = -1 * net(x + delta)

        loss_penalty = LAMBDA2 * (torch.norm(delta, p=2, dim=1) ** 2)

        loss_delta = (loss_objective.squeeze() + loss_penalty.squeeze()).mean()

        gradient = torch.autograd.grad(loss_delta, delta, retain_graph=True)[0]
        if normalize:
            new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1), keepdim=True))
        else:
            new_delta = delta - stepsize * gradient 
        if LAMBDA1 > 0:
            new_delta = soft_threshold(new_delta, LAMBDA1)
        delta = new_delta

    delta = delta.detach()

    return delta

def loss_wass(net, fake, real):
    real_prediction = net(real)
    fake_prediction = net(fake)
    loss = -1 * (torch.mean(real_prediction) - torch.mean(fake_prediction))
    # print("Network Loss: {:.4f}, Real_Predict: {:.4f}, Fake_Predict: {:.4f}".format(loss.item(), real_prediction.mean().item(), fake_prediction.mean().item()))
    return loss

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.abs(image_numpy)
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def visual(tensor, filename):
    tmp = tensor2im(tensor)
    save_image(tmp, filename)

def soft_threshold(delta, threshold):
    larger = delta > threshold
    smaller = delta < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    delta = delta * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    delta = delta + subtracted + added

    return delta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1', type=float, default=5e-3)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--pad_size', type=int, default=56)

    opt = parser.parse_args()

    dataset_x = mnist_digit_dataset(1)
    dataset_y = mnist_digit_dataset(9)

    BATCHSIZE = 128
    synX = DataLoader(dataset_x, batch_size=BATCHSIZE, shuffle=True, pin_memory=True)
    synY = DataLoader(dataset_y, batch_size=BATCHSIZE, shuffle=True, pin_memory=True)

    LAMBDA2 = opt.l2
    LAMBDA1 = opt.l1
    SIZE_DATA = 28
    SIZE_TOTAL = opt.pad_size
    SIZE_PAD = int((SIZE_TOTAL - SIZE_DATA) / 2.0)
    num_features = SIZE_TOTAL ** 2
    EPOCHS = 40
    SAVE_PATH = ...
    IMG_SAVEPATH = os.path.join(SAVE_PATH, 'img')
    NET_SAVEPATH = os.path.join(SAVE_PATH, 'net')

    os.makedirs(IMG_SAVEPATH, exist_ok=True)
    os.makedirs(NET_SAVEPATH, exist_ok=True)
    num_critics = 5
    stepsize = .1

    net = MLP(num_features)
    net = add_sn(net).cuda()

    optim = Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS)
    sparse_record = []

    for epoch in range(EPOCHS):
        pbar = tqdm(range(len(synX)))
        for itr in pbar:
            start_time = time.time()

            x, _ = next(iter(synX))
            y, _ = next(iter(synY))

            if SIZE_TOTAL > 28:
                x_noisy = torch.rand((BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL))
                y_noisy = torch.rand((BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL))
                x_noisy[:, :, SIZE_PAD:-1 * SIZE_PAD, SIZE_PAD:-1 * SIZE_PAD] = x
                y_noisy[:, :, SIZE_PAD:-1 * SIZE_PAD, SIZE_PAD:-1 * SIZE_PAD] = y

                x, y = x_noisy.cuda(), y_noisy.cuda()
            
            else:
                x, y = x.cuda(), y.cuda()
            x, y = x.reshape(BATCHSIZE, -1), y.reshape(BATCHSIZE, -1)
            
            prepare_time = time.time() - start_time

            delta = delta_l2(net, x, stepsize, LAMBDA2, LAMBDA1, True)


            for j in range(num_critics):

                loss = loss_wass(net, x+delta, y)

                optim.zero_grad()
                loss.backward()
                optim.step()


            process_time = time.time() - prepare_time
            pbar.set_description("Epoch: {}/{}, Eff: {:.2f}, Diff: {:.4f}, Norm: {:.4f}".format(
                epoch, EPOCHS, process_time / (process_time + prepare_time) , -1 * loss.item(), torch.norm(delta, p=2, dim=(1), keepdim=True).mean()
            ))
        
        sparse_delta = delta.clone()
        sparse_delta = sparse_delta.reshape((BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL))
        sparse_delta[:, :, SIZE_PAD:-1 * SIZE_PAD, SIZE_PAD:-1 * SIZE_PAD] = 0
        sparsity = ((sparse_delta == 0).sum() - (BATCHSIZE * SIZE_DATA ** 2)) / (BATCHSIZE * (num_features - SIZE_DATA ** 2))
        print("Pad_Sparse: {}".format(sparsity))
        sparse_record.append(sparsity)
        torch.save(sparse_record, os.path.join(NET_SAVEPATH, 'spar.pth'))

        trans = x + delta

        visual((x).reshape(BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL), os.path.join(IMG_SAVEPATH, '{}_sour.png'.format(epoch)))
        visual((delta).reshape(BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL), os.path.join(IMG_SAVEPATH, '{}_delta.png'.format(epoch)))
        # visual((grad).reshape(BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL), os.path.join(IMG_SAVEPATH, '{}_grad.png'.format(epoch)))
        visual((trans).reshape(BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL), os.path.join(IMG_SAVEPATH, '{}_trans.png'.format(epoch)))
        visual((y).reshape(BATCHSIZE, 1, SIZE_TOTAL, SIZE_TOTAL), os.path.join(IMG_SAVEPATH, '{}_target.png'.format(epoch)))

        scheduler.step()
    
    torch.save(net.state_dict(), os.path.join(NET_SAVEPATH, 'net.pth'))
