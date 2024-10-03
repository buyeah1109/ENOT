import torch
import itertools
import matplotlib.pyplot as plt
import networks
import os

from torch.nn.parameter import Parameter
from tqdm import tqdm
import torchvision
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
from PIL import Image
from torchvision.models import resnet18, inception_v3, mobilenet_v2, resnet50, ResNet50_Weights
import glob
from prefetch_generator import BackgroundGenerator
import time
from efficientnet_pytorch import EfficientNet
import argparse
from torch.autograd import Variable

mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))    

    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return spectral_norm(m)
    else:
        return m

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
        # image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0, 0, 255)  # post-processing: tranpose and scaling
        image_numpy = np.abs(image_numpy)
        image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0)) * 255.0, 0, 255)  # post-processing: tranpose and scaling

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

def soft_threshold(delta, threshold):
    larger = delta > threshold
    smaller = delta < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    delta = delta * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    delta = delta + subtracted + added

    return delta

def mnist_digit_dataset(digit):
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

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))[:1000]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        label = 0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def folder_dataloader(dir, batchsize, flip=True, resize=224):
    transforms = []
    if flip:
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    
    base_transform = [torchvision.transforms.Resize([resize, resize]),
                    torchvision.transforms.ToTensor()]
                    # torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    transform = torchvision.transforms.Compose(transforms + base_transform)  
    dataset = CustomImageDataset(img_dir=dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True)
    return dataloader

def loss_wass_log(net, fake, real):
    real_prediction = net(real)
    fake_prediction = net(fake)
    loss = -1 * (torch.mean(torch.log(real_prediction + 1e-16)) + torch.mean(torch.log(1 - fake_prediction + 1e-16)))
    print("Network Loss: {:.4f}, Real_Predict: {:.4f}, Fake_Predict: {:.4f}".format(loss.item(), real_prediction.mean().item(), fake_prediction.mean().item()))
    return loss

def loss_wass(net, fake, real):
    real_prediction = net(real)
    fake_prediction = net(fake)
    loss = -1 * (torch.mean(real_prediction) - torch.mean(fake_prediction))
    print("Network Loss: {:.4f}, Real_Predict: {:.4f}, Fake_Predict: {:.4f}".format(loss.item(), real_prediction.mean().item(), fake_prediction.mean().item()))
    return loss

def loss_transport_l2(net, lambda2, source, delta):
    loss_objective = torch.log(1 - net(source + delta))
    loss_delta = lambda2 * (torch.norm(delta, p=2, dim=(1, 2, 3)) ** 2) 

    return loss_objective.squeeze() + loss_delta.squeeze()

def optimize_network(net, optimizer, real, fake, clip=0, use_log=True, patchgan=False):
    if patchgan:

        loss_func = torch.nn.MSELoss()

        real_prediction = net(real)
        fake_prediction = net(fake)
        real_target = torch.tensor([1.0]).expand_as(real_prediction).cuda()
        fake_target = torch.tensor([0.0]).expand_as(fake_prediction).cuda()

        loss_real = loss_func(real_prediction, real_target)
        loss_fake = loss_func(fake_prediction, fake_target)
        loss = (loss_real + loss_fake) * 0.5
        print("Network Loss: {:.4f}, Real: {:.4f}, Fake: {:.4f}".format(loss.item(), real_prediction.mean().item(), fake_prediction.mean().item()))

    elif use_log:
        loss = loss_wass_log(net, fake, real)
    else:
        loss = loss_wass(net, fake, real)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if clip > 0:
        for p in net.parameters():
            p.data.clamp_(-1 * clip, clip)

def save(idx, save_path, delta, transfer, source):

    save_delta = os.path.join(save_path , '{}_delta.png'.format(idx))
    save_source = os.path.join(save_path , '{}_source.png'.format(idx))
    save_trans = os.path.join(save_path , '{}_transfered.png'.format(idx))
    save_mask = os.path.join(save_path , '{}_masked.png'.format(idx))

    img_delta = tensor2im(delta)
    img_trans = tensor2im(transfer)
    img_source = tensor2im(source)
    img_masked = tensor2im(source * (delta > 0))

    save_image(img_delta, save_delta)
    save_image(img_trans, save_trans)
    save_image(img_source, save_source)
    save_image(img_masked, save_mask)


def init_delta(size, ds=False):
    delta = torch.zeros(size)
    delta = delta.cuda()
    delta = Parameter(delta, requires_grad = True)

    if ds:
        delta_sparse = torch.zeros(size)
        delta_sparse = delta_sparse.cuda()
        delta_sparse = Parameter(delta_sparse, requires_grad = True)

    if ds:
        return delta, delta_sparse
    return delta

def optimize_delta(net, data, use_log=False, patchgan=False):
    source_real = data

    delta = init_delta(source_real.shape)

    itr_bar = tqdm(range(ITR_DELTA))
    # optimizer = torch.optim.Adam([delta], lr=5e-3)

    for i in enumerate(itr_bar):
        if patchgan:
            prediction = net(source_real + delta)
            loss_func = torch.nn.MSELoss()

            real_target = torch.tensor([1.0]).expand_as(prediction).cuda()

            loss_objective = loss_func(prediction, real_target)

        elif use_log:
            loss_objective = torch.log(1 - net(source_real + delta) + 1e-16).mean()
        else:
            loss_objective = -1 * net(source_real + delta).mean()

        loss_penalty = LAMBDA2 * (torch.norm(delta, p=2, dim=(1, 2, 3)) ** 2).squeeze().mean()

        # print(loss_objective.shape, loss_penalty.shape, loss_penalty_sparse.shape)
        # assert loss_objective.shape == loss_penalty.shape and loss_objective.shape == loss_penalty_sparse.shape
        loss_delta = (loss_objective + loss_penalty).mean()

        gradient = torch.autograd.grad(loss_delta, delta)[0]
        stepsize = 5
        new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1, 2, 3), keepdim=True))
        delta = new_delta

        itr_bar.set_description("loss: {:.2f}, floss: {:.6f}, ploss: {:.6f}".format(
            loss_delta.item(), loss_objective.mean().item(), loss_penalty.mean().item() / LAMBDA2
        ))
    
    return delta.detach()

def optimize_delta_l2(net, source, target):
    target_feature = net(target)

    loss_func = torch.nn.MSELoss()
    delta, delta_sparse = init_delta(source.shape)

    itr_bar = tqdm(range(ITR_DELTA))
    for i in enumerate(itr_bar):

        loss_objective = loss_func(target_feature, net(source + delta + delta_sparse))   

        loss_penalty = LAMBDA2 * (torch.norm(delta, p=2, dim=(1, 2, 3)) ** 2).mean()
        loss_penalty_sparse = LAMBDA2_sparse * (torch.norm(delta_sparse, p=2, dim=(1, 2, 3)) ** 2).mean()

        assert loss_objective.shape == loss_penalty.shape and loss_objective.shape == loss_penalty_sparse.shape
        loss_delta = loss_objective + loss_penalty + loss_penalty_sparse

        gradient = torch.autograd.grad(loss_delta, delta, retain_graph=True)[0]
        stepsize = 1
        new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1, 2, 3), keepdim=True))
        # new_delta = delta - stepsize * gradient
        delta = new_delta

        gradient_sparse = torch.autograd.grad(loss_delta, delta_sparse)[0]
        stepsize_sparse = 1
        new_delta_sparse = delta_sparse - stepsize_sparse * (gradient_sparse / torch.norm(gradient_sparse, p=2, dim=(1, 2, 3), keepdim=True))
        # new_delta_sparse = delta_sparse - stepsizew_sparse * gradient_sparse 
        new_delta_sparse = soft_threshold(new_delta_sparse, LAMBDA1)
        delta_sparse = new_delta_sparse

        itr_bar.set_description("loss: {:.2f}, floss: {:.6f}, ploss: {:.6f}, ploss_sparse: {:.6f}".format(
            loss_delta.item(), loss_objective.mean().item(), loss_penalty.mean().item() / LAMBDA2, loss_penalty_sparse.mean().item() / LAMBDA2_sparse
        ))
    
    return delta.detach(), delta_sparse.detach()


def train(net, use_log=True, mnist=False, patchgan=False):
    net.train()
    torch.cuda.set_device(gpu_ids[0])

    # SAVE_PATH = os.path.join(SAVE_PATH, 'sparse_{}'.format(LAMBDA1))
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
    os.makedirs(NETWORK_SAVE_PATH, exist_ok=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR_NETWORK, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    if mnist:
        dataloader_s = torch.utils.data.DataLoader(mnist_digit_dataset(4), batch_size = 200, shuffle=True)
        dataloader_t = torch.utils.data.DataLoader(mnist_digit_dataset(8), batch_size = 200, shuffle=True)

    else:

        dataloader_s = folder_dataloader(..., batchsize=64)
        dataloader_t = folder_dataloader(..., batchsize=64)

    for epoch_num in range(100):
        start_time = time.time()

        for j, (source, label) in enumerate(dataloader_s):
            if mnist:
                source = source.repeat(1, 3, 1, 1)

            torch.cuda.set_device(gpu_ids[0])
            source = source.cuda()

            if epoch_num >= WARMUP_EPOCHS:
                delta = optimize_delta(net, source, use_log=use_log, patchgan=patchgan)
                # delta = torch.zeros_like(source)
            else:
                delta = torch.zeros_like(source)
                # delta_sparse = torch.zeros_like(source)

            target, _ = next(iter(dataloader_t))
            if mnist:
                target = target.repeat(1, 3, 1, 1)
            target = target.cuda()
            transferred = source + delta.detach()
            for repeat in range(1):
                optimize_network(net, optimizer=optimizer, real=target, fake=transferred, use_log=use_log, patchgan=patchgan)
        
        if epoch_num > WARMUP_EPOCHS:
            save(epoch_num, IMAGE_SAVE_PATH, delta, transfer=transferred, source=source)

        if (epoch_num) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(NETWORK_SAVE_PATH, '{}.pth'.format(epoch_num + 1)))

        scheduler.step()
        print("epoch: {}, time: {:.2f}s, LR: {:.6f}".format(epoch_num, time.time() - start_time, scheduler.get_last_lr()[0]))

def train_vit(net):
    net.train()
    torch.cuda.set_device(gpu_ids[0])

    # SAVE_PATH = os.path.join(SAVE_PATH, 'sparse_{}'.format(LAMBDA1))
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
    os.makedirs(NETWORK_SAVE_PATH, exist_ok=True)

    dataloader_s = folder_dataloader(..., batchsize=16)
    dataloader_t = folder_dataloader(..., batchsize=16)

    head = torch.nn.Sequential(
        torch.nn.Linear(768, 1),
        torch.nn.Sigmoid()
    )
    head.cuda()
    optimizer_net = torch.optim.Adam(net.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    for epoch in range(100):

        for j, (source, label) in enumerate(dataloader_s):

            torch.cuda.set_device(gpu_ids[0])
            source = source.cuda()
            target, _ = next(iter(dataloader_t))
            target = target.cuda()

            source_feature = net.forward_features(source)
            target_feature = net.forward_features(target)

            target = head(target_feature)
            source = head(source_feature)
            loss = -1 * (torch.log(target).mean() + torch.log(1 - source).mean())

            optimizer_net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer_net.step()
            optimizer.step()

            print("epoch: {}, loss: {:.6f}, source: {:.6f}, target: {:.6f}".format(epoch, loss.item(), source.mean().item(), target.mean().item()))

        torch.save(net.state_dict(), os.path.join(NETWORK_SAVE_PATH, 'vit_{}.pth'.format(epoch + 1)))
        torch.save(head.state_dict(), os.path.join(NETWORK_SAVE_PATH, '{}.pth'.format(epoch + 1)))

seed = 3
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
BATCHSIZE = 1
LAMBDA2 = 1e-6
LAMBDA2_sparse = 100
LAMBDA1 = 0

ITR_DELTA = 100

LR_NETWORK = 1e-3
LR_DELTA = 1

NUM_EXAMPLES = 2
SAVE_PATH = ...
IMAGE_SAVE_PATH = os.path.join(SAVE_PATH, 'sample')
NETWORK_SAVE_PATH = os.path.join(SAVE_PATH, 'network')
WARMUP_EPOCHS = 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--mnist', type=bool)
    parser.add_argument('--use_log', type=bool)
    parser.add_argument('--patchgan', type=bool)

    opt = parser.parse_args()

    use_log = opt.use_log
    mnist = opt.mnist
    patchgan = opt.patchgan
    LAMBDA1 = opt.l1
    gpu_ids = [0]

    import vit_model
    net = vit_model.load_vit_base()
    net = torch.nn.Sequential(
        net,
        torch.nn.Linear(1000, 1),
        torch.nn.Sigmoid()
    )
    net = add_sn(net)

    net.cuda()
    
    train(net, use_log=use_log, mnist=mnist, patchgan=patchgan)