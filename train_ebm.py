#######################################
# ## TRAIN EBM USING IMAGE DATASET ## #
#######################################

import torch as t
import torchvision as tv
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

import json
import datetime

from nets import EBM
from utils import setup_exp, import_data

# json file with experiment config
CONFIG_FILE = './config_train_ebm/cifar10_ebm.json'


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)
# directory for experiment results
exp_dir = config['exp_dir'] + '_' + datetime.datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p') + '_/'
# setup folders, save code, set seed and get device
setup_exp(exp_dir, config['seed'], ['checkpoints', 'shortrun', 'longrun', 'plots'],
          ['train_ebm.py', 'nets.py', 'utils.py', CONFIG_FILE])

print('Setting up network and optimizer...')
# set up network
ebm = EBM().cuda()
# set up adam optimizer for first learning phase
optim = t.optim.Adam(ebm.parameters(), lr=config['lr_adam'])

print('Processing data...')
# set up training data
data = import_data(config['data_type'], True, False)[0]
q = next(iter(DataLoader(data, len(data), num_workers=0)))[0].cuda()

if config['shortrun_init'] == 'persistent':
    # initialize persistent images from noise (one persistent image for each data image)
    s_t_0 = 2 * t.rand_like(q) - 1


################################
# ## FUNCTIONS FOR LEARNING ## #
################################

# sample batch from given array of images
def sample_image_set(image_set):
    rand_inds = t.randperm(image_set.shape[0])[0:config['batch_size']]
    return image_set[rand_inds], rand_inds

# sample positive images from dataset distribution q (add noise to ensure min sd is at least langevin noise sd)
def sample_q():
    x_q = sample_image_set(q)[0]
    return x_q + config['data_epsilon'] * t.randn_like(x_q)

# initialize and update images with langevin dynamics to obtain samples from finite-step MCMC distribution s_t
def sample_s_t(L, init_type, update_s_t_0=True):
    # get initial mcmc states for langevin updates ("persistent", "data", "uniform", "gaussian")
    def sample_s_t_0():
        if init_type == 'persistent':
            return sample_image_set(s_t_0)
        elif init_type == 'data':
            return sample_q(), None
        elif init_type == 'uniform':
            noise_image = 2 * t.rand([config['batch_size'], 3, 32, 32]) - 1
            return noise_image.cuda(), None
        elif init_type == 'gaussian':
            noise_image = t.randn([config['batch_size'], 3, 32, 32])
            return noise_image.cuda(), None
        else:
            raise RuntimeError('Invalid "init_type" ("persistent", "data", "uniform", "gaussian")')

    # initialize MCMC samples
    x_s_t_0, s_t_0_inds = sample_s_t_0()

    # iterative langevin updates of MCMC samples
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1).cuda()  # variable to record average gradient magnitude
    for ell in range(L):
        # gradient from energy
        u_prime = t.autograd.grad(ebm(x_s_t).sum(), [x_s_t])[0]
        # langevin update
        x_s_t.data += - u_prime + config['epsilon'] * t.randn_like(x_s_t)
        # record langevin gradient magnitude
        r_s_t += u_prime.view(u_prime.shape[0], -1).norm(dim=1).mean()

    if init_type.startswith('persistent') and update_s_t_0:
        # update persistent image bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L


#############################
# ## DIAGNOSTIC PLOTTING ## #
#############################

# plot diagnostics for learning
def plot_diagnostics(en_diffs, grad_mags, exp_dir, fontsize=10):
    # axis tick size
    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    fig = plt.figure()

    def plot_en_diff_and_grad_mag():
        # energy difference
        ax = fig.add_subplot(211)
        ax.plot(en_diffs)
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Energy Difference', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$d_{s_t}$', fontsize=fontsize)
        # mean langevin gradient
        ax = fig.add_subplot(212)
        ax.plot(grad_mags)
        ax.set_title('Average Langevin Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$r_{s_t}$', fontsize=fontsize)

    # make diagnostic plot and save
    plot_en_diff_and_grad_mag()
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.savefig(exp_dir + 'diagnosis_plot.pdf', format='pdf')
    plt.close()

# save images to file
def plot_ims(p, x):
    tv.utils.save_image(t.clamp(x, -1., 1.), p, normalize=True, nrow=int(config['batch_size'] ** 0.5))


#######################
# ## TRAINING LOOP ## #
#######################

# containers for diagnostic records
d_s_t_record = t.zeros([0]).cuda()  # energy difference between positive and negative samples
r_s_t_record = t.zeros([0]).cuda()  # average image gradient magnitude along Langevin path

print('Training has started.')
for i in range(config['num_train_batches']):
    # switch to SGD optimizer at a certain step in training
    if (i + 1) == config['optimizer_switch_step']:
        print('{:>6d}   Switching from ADAM to SGD optimizer for network weight updates'.format(i + 1))
        optim = t.optim.SGD(ebm.parameters(), lr=config['lr_sgd'] * (config['epsilon'] ** 2) / 2)

    # obtain positive and negative samples
    x_q = sample_q()
    x_s_t, r_s_t = sample_s_t(L=config['num_shortrun_steps'], init_type=config['shortrun_init'])

    # calculate ML computational loss
    d_s_t = (ebm(x_q).mean() - ebm(x_s_t).mean()) / ((config['epsilon']**2) / 2)
    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # record diagnostics
    d_s_t_record = t.cat((d_s_t_record, d_s_t.detach().view(1)), 0)
    r_s_t_record = t.cat((r_s_t_record, r_s_t.view(1)), 0)

    # print and save learning info
    if (i + 1) == 1 or (i + 1) % config['log_freq'] == 0:
        print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(i+1, d_s_t.detach().data, r_s_t))
        # visualize synthesized images
        plot_ims(exp_dir + 'shortrun/x_s_t_{:>06d}.png'.format(i+1), x_s_t)
        # save network weights
        t.save(ebm.state_dict(), exp_dir + 'checkpoints/net_{:>06d}.pth'.format(i+1))
        # save optim
        t.save(optim.state_dict(), exp_dir + 'checkpoints/optim.pth')
        # save learning record
        t.save({'ens': d_s_t_record.cpu(), 'grads': r_s_t_record.cpu()}, exp_dir + 'checkpoints/records.pth')
        if config['shortrun_init'] == 'persistent':
            # visualize batch of persistent states and save persistent bank
            plot_ims(exp_dir + 'shortrun/x_s_t_0_{:>06d}.png'.format(i+1), s_t_0[0:config['batch_size']])
            t.save({'persistent_states': s_t_0.cpu()}, exp_dir + 'checkpoints/persistent_states.pth')
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(d_s_t_record.cpu(), r_s_t_record.cpu(), exp_dir + 'plots/')

    # sample longrun chains to diagnose model steady-state
    if config['log_longrun'] and (i+1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i+1, config['num_longrun_steps']))
        x_p_theta = sample_s_t(L=config['num_longrun_steps'], init_type=config['longrun_init'], update_s_t_0=False)[0]
        plot_ims(exp_dir + 'longrun/longrun_{:>06d}.png'.format(i+1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i+1))
