from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Stochastic_Gradient_HMC_SA.model import BNN_cat
from src.utils import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on MNIST with Scale-adapted Stochastic Gradient HMC')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=250,
                    help='How many epochs to train. Default: 250.')
parser.add_argument('--sample_freq', type=int, nargs='?', action='store', default=2,
                    help='How many epochs pass between saving samples. Default: 2.')
parser.add_argument('--burn_in', type=int, nargs='?', action='store', default=20,
                    help='How many epochs to burn in for?. Default: 20.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-2,
                    help='learning rate. I recommend 1e-2. Default: 1e-2.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='SGHMC_models',
                    help='Where to save learnt weights and train vectors. Default: \'SGHMC_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='SGHMC_results',
                    help='Where to save learnt training plots. Default: \'SGHMC_results\'.')
args = parser.parse_args()



# Where to save models weights
models_dir = args.models_dir
# Where to save plots and error, accuracy vectors
results_dir = args.results_dir

mkdir(models_dir)
mkdir(results_dir)
# ------------------------------------------------------------------------------------------------------
# train config
NTrainPointsMNIST = 60000
batch_size = 256
nb_epochs = args.epochs
log_interval = 1
nb_its_dev = log_interval
flat_ims=True
# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# load data

# data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

use_cuda = torch.cuda.is_available()

trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)

if use_cuda:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)

else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                            num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

lr = args.lr
########################################################################################


net = BNN_cat(NTrainPointsMNIST, lr=lr, cuda=use_cuda, grad_std_mul=20)


## weight saving parameters #######
burn_in = args.burn_in
sim_steps = args.sample_freq
N_saves=100
resample_its = 50
resample_prior_its = 15
re_burn = 1e8
###################################

## ---------------------------------------------------------------------------------------------------------------------



# net dims
epoch = 0
it_count = 0
## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c', '\nTrain:')

print('  init cost variables:')
cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)
cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
best_cost = np.inf
best_err = np.inf

tic0 = time.time()
for i in range(epoch, nb_epochs):
    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0
    for x, y in trainloader:

        if flat_ims:
            x = x.view(x.shape[0], -1)

        cost_pred, err = net.fit(x, y, burn_in=(i % re_burn < burn_in),
                                 resample_momentum=(it_count % resample_its == 0),
                                 resample_prior=(it_count % resample_prior_its == 0))
        it_count += 1
        err_train[i] += err
        cost_train[i] += cost_pred
        nb_samples += len(x)

    cost_train[i] /= nb_samples
    err_train[i] /= nb_samples
    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, nb_epochs, cost_train[i], err_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))
    net.update_lr(i)

    # ---- save weights
    if i % re_burn >= burn_in and i % sim_steps == 0:
        net.save_sampled_net(max_samples=N_saves)

    # ---- dev
    if i % nb_its_dev == 0:
        nb_samples = 0
        for j, (x, y) in enumerate(valloader):
            if flat_ims:
                x = x.view(x.shape[0], -1)

            cost, err, probs = net.eval(x, y)

            cost_dev[i] += cost
            err_dev[i] += err
            nb_samples += len(x)

        cost_dev[i] /= nb_samples
        err_dev[i] /= nb_samples

        cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
        if err_dev[i] < best_err:
            best_err = err_dev[i]
            cprint('b', 'best test error')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

## SAVE WEIGHTS
net.save_weights(models_dir + '/state_dicts.pkl')


## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its

# fig cost vs its
textsize = 15
marker = 5

plt.figure(dpi=100)
fig, ax1 = plt.subplots()
ax1.plot(range(0, nb_epochs, nb_its_dev), np.clip(cost_dev[::nb_its_dev], a_min=-5, a_max=5), 'b-')
ax1.plot(np.clip(cost_train, a_min=-5, a_max=5), 'r--')
ax1.set_ylabel('Cross Entropy')
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
ax = plt.gca()
plt.title('classification costs')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')
plt.savefig(results_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(dpi=100)
fig2, ax2 = plt.subplots()
ax2.set_ylabel('% error')
ax2.semilogy(range(0, nb_epochs, nb_its_dev), err_dev[::nb_its_dev], 'b-')
ax2.semilogy(err_train, 'r--')
ax2.set_ylim(top=1, bottom=1e-3)
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
ax = plt.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(textsize)
    item.set_weight('normal')
plt.savefig(results_dir + '/err.png', bbox_extra_artists=(lgd,), box_inches='tight')
plt.show()
