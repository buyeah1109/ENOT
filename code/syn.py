import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, AdamW
import matplotlib.pyplot as plt
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.parameter import Parameter
from tqdm import tqdm
import time
import argparse
from scipy import linalg
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class MLP(nn.Module):
    def __init__(self, num_features, depth):
        super(MLP, self).__init__()

        # self.hidden_feature = int(num_features / 100)
        self.hidden_feature = 50
        self.hidden_nums = depth

        self.linear_map = nn.Linear(num_features, self.hidden_feature)

        self.hiddens = []
        for i in range(self.hidden_nums):
            self.hiddens.append(torch.nn.Sequential(nn.Linear(self.hidden_feature, self.hidden_feature, device=0),
                                                     nn.Sigmoid(),
                                                      nn.Linear(self.hidden_feature, self.hidden_feature, device=0)))

        self.linear_f = nn.Linear(self.hidden_feature, 1)


        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear_map(x)
        x = self.act(x)

        for hidden in self.hiddens:
            inputs = x
            x = hidden(x)
            x = self.act(x)
            x = inputs + x

        x = self.linear_f(x)

        return x

class Quad(nn.Module):
    def __init__(self, num_features, lambda2):
        super(Quad, self).__init__()
        self.para = Parameter(torch.zeros((num_features, 1)), requires_grad=True)
        self.lambda2 = lambda2
    
    def forward(self, x):
        output = 0.5 * self.lambda2 * (torch.norm(x, p=2, dim=1) ** 2) + torch.matmul(x, self.para).squeeze()
        return output


class SynDataset(Dataset):
    def __init__(self, num, dim, seed, mask_sparsity):
        self.num = num
        self.dim = dim
        self.seed = seed
        self.datas, self.bias = generate_syn_data(num, dim, seed, mask_sparsity)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        syn = self.datas[idx]
        return syn
    
    def get_bias(self):
        return self.bias

class SynGMDataset(Dataset):
    def __init__(self, num, dim, seed, sparse_dim_value=10, eps=0.02, std=0.05):
        self.num = num
        self.dim = dim
        self.seed = seed

        self.datas = generate_syn_GM_data(num, dim, seed, sparse_dim_value, eps, std)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        syn = self.datas[idx]
        return syn
    
    # def get_bias(self):
    #     return self.bias
    

def generate_syn_GM_data(num, dim, seed, sparse_dim_value, eps, std):
    torch.manual_seed(seed)
    samples = torch.rand(size=(num-1, 1))
    means = torch.FloatTensor([sparse_dim_value] + [eps for i in range(dim-1)])
    for sample in samples:

        if sample <= 0.5:
            sample_dist_mean = torch.FloatTensor([sparse_dim_value] + [eps for i in range(dim-1)])

        else:
            sample_dist_mean = torch.FloatTensor([-sparse_dim_value] + [-eps for i in range(dim-1)])

        means = torch.vstack([means, sample_dist_mean])

    print(f"length: {len(sample_dist_mean)}")
    return torch.normal(means, std)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    # print('calculate diff')
    diff = mu1 - mu2

    # Product might be almost singular
    # print('calculate covmean')
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=2e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # print('calculate tr_covmean')
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def generate_syn_data(num, dim, seed, mask_sparsity):
    torch.manual_seed(seed)
    bias = torch.randn((1, dim))
    if mask_sparsity > 0:
        mask = []
        for i in range(dim):
            if torch.rand(1) > mask_sparsity:
                mask.append(1)
            else:
                mask.append(0)

        mask = torch.FloatTensor(mask)
        bias = bias * mask
    print("Bias Sparsity: {:4f}".format((bias==0).sum()))
    dataset = 0.1 * torch.randn((num, dim)) + bias
    return dataset, bias


def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))    

    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return spectral_norm(m)
    else:
        return m

def loss_wass(net, fake, real):
    real_prediction = net(real)
    fake_prediction = net(fake)
    loss = -1 * (torch.mean(real_prediction) - torch.mean(fake_prediction))
    return loss

def soft_threshold(delta, threshold):
    larger = delta > threshold
    smaller = delta < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    delta = delta * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    delta = delta + subtracted + added

    return delta

def delta_l2(net, x, LAMBDA2, LAMBDA1=0, normalize=True):
    # from x to y
    delta = torch.zeros(x.shape)
    delta = delta.cuda()
    delta = Parameter(delta, requires_grad = True)
        
    for i in range(100):
        loss_objective = -1 * net(x + delta)
        loss_penalty = LAMBDA2 * (torch.norm(delta, p=2, dim=1) ** 2)
        loss_delta = (loss_objective.squeeze() + loss_penalty.squeeze()).mean()

        gradient = torch.autograd.grad(loss_delta, delta, retain_graph=True)[0]
        stepsize = 1

        if normalize:
            new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1), keepdim=True))
        else:
            new_delta = delta - stepsize * gradient 
        if LAMBDA1 > 0:
            new_delta = soft_threshold(new_delta, LAMBDA1)
        delta = new_delta

    delta = delta.detach()

    return delta

def gm_nll(x, dim, mean1, mean2, std):
    gm1 = GaussianMixture(n_components=2)

    gm1.precisions_cholesky_ = np.array([1 / (std**2) * np.identity(dim), 1 / (std**2) * np.identity(dim)])
    gm1.means_ = np.array([mean1, mean2])
    gm1.weights_ = np.array([.5, .5])  

    return -gm1.score(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--std', type=float, default=1)
    parser.add_argument('--train_itr', type=int)

    opt = parser.parse_args()

    torch.manual_seed(opt.seed)

    num_features = opt.dim
    num_samples = opt.num_samples
    sparse_dim_value = opt.gamma
    eps = opt.eps
    std = opt.std

    dataset_x = SynGMDataset(num_samples, num_features, seed=3, sparse_dim_value=sparse_dim_value, eps=eps, std=std)
    dataset_y = SynGMDataset(num_samples, num_features, seed=42, sparse_dim_value=-sparse_dim_value, eps=eps, std=std)

    X_train, X_test = torch.utils.data.random_split(dataset_x, [1000, 1000])
    Y_train, Y_test = torch.utils.data.random_split(dataset_y, [1000, 1000])


    synX_train = DataLoader(X_train, batch_size=1000, shuffle=True, pin_memory=True)
    synX_test = DataLoader(X_test, batch_size=1000, shuffle=True, pin_memory=True)

    synY_train = DataLoader(Y_train, batch_size=1000, shuffle=True, pin_memory=True)
    synY_test = DataLoader(Y_test, batch_size=1000, shuffle=True, pin_memory=True)

    sparse_levels = [0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    LAMBDA2s = [1e-8]
    target_mean1 = np.array([-sparse_dim_value] + [eps for j in range(num_features - 1)])
    target_mean2 = np.array([sparse_dim_value] + [-eps for j in range(num_features - 1)])


    for lambda2 in LAMBDA2s:
        for sparse in sparse_levels:
            print("L2: {}, L1: {}, dim: {}".format(lambda2, sparse, opt.dim))

            max_itr = opt.train_itr
            itr = 1
            cnt = 0
            norm_b = 100

            net = MLP(num_features, opt.depth).cuda()
            net = add_sn(net).cuda()

            optim = SGD(net.parameters(), lr=1e-2, momentum=.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_itr)
            record = []

            pbar = tqdm(range(max_itr))
            res = 0
            for itr in pbar:
                start_time = time.time()

                x = next(iter(synX_train))
                y = next(iter(synY_train))
                cnt += 1

                x, y = x.cuda(), y.cuda()

                prepare_time = time.time() - start_time

                delta = delta_l2(net, x, lambda2, sparse, True)

                loss = loss_wass(net, x+delta, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                process_time = time.time() - prepare_time
                pbar.set_description("Eff: {:.2f}, Diff: {:.4f}, Dist: {:.2f}".format(
                    process_time / (process_time + prepare_time) , -1 * loss.item(), gm_nll((x+delta).cpu().numpy(), num_features, target_mean1, target_mean2, std)))
            
            mean_t1 = []
            nll = 0
            
            x = next(iter(synX_test))
            x = x.cuda()
            delta = delta_l2(net, x, lambda2, sparse, True)
            transferred = (x + delta).cpu().numpy()

            res = gm_nll((x+delta).cpu().numpy(), num_features, target_mean1, target_mean2, std)
            print(f'nll: {res}')
