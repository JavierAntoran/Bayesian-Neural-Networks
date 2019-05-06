from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.KF_Laplace.model import *
from src.KF_Laplace.hessian_operations import chol_scale_invert_kron_factor

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on MNIST with MC Dropout Variational Inference')

parser.add_argument('--weight_decay', type=float, nargs='?', action='store', default=0.01,
                    help='Specify the precision of an isotropic Gaussian prior. Default: 0.01')
parser.add_argument('--hessian_diag_sig', type=float, nargs='?', action='store', default=0.15,
                    help='Specify Gaussian prior std for a diagonal term that is added to the approximate hessian. Default: 0.15.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=10,
                    help='How many epochs to train. Default: 10.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='KFLaplace_models',
                    help='Where to save learnt weights, train vectors and Hessian params. Default: \'KFLaplace_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='KFLaplace_results',
                    help='Where to save learnt training plots. Default: \'KFLaplace_results\'.')
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
batch_size = 128
nb_epochs = args.epochs
log_interval = 1


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
prior_sig = np.sqrt(1/args.weight_decay)
########################################################################################



net = KBayes_Net(lr=lr, channels_in=1, side_in=28, cuda=use_cuda, classes=10, n_hid=1200, batch_size=batch_size, prior_sig=prior_sig)

## ---------------------------------------------------------------------------------------------------------------------
# train
epoch = 0
cprint('c', '\nTrain:')

print('  init cost variables:')
pred_cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)

cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
best_err = np.inf

nb_its_dev = 1

tic0 = time.time()
for i in range(epoch, nb_epochs):

    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0

    for x, y in trainloader:
        cost_pred, err = net.fit(x, y)

        err_train[i] += err
        pred_cost_train[i] += cost_pred
        nb_samples += len(x)

    pred_cost_train[i] /= nb_samples
    err_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, nb_epochs, pred_cost_train[i], err_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))


    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        for j, (x, y) in enumerate(valloader):
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
            net.save(models_dir+'/theta_best.dat')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

net.save(models_dir+'/theta_last.dat')

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c', '\nRESULTS:')
nb_parameters = net.get_nb_parameters()
best_cost_dev = np.min(cost_dev)
best_cost_train = np.min(pred_cost_train)
err_dev_min = err_dev[::nb_its_dev].min()

print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
print('  err_dev: %f' % (err_dev_min))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))

## Save results for plots
np.save(results_dir + '/cost_train.npy', pred_cost_train)
np.save(results_dir + '/cost_dev.npy', cost_dev)
np.save(results_dir + '/err_train.npy', err_train)
np.save(results_dir + '/err_dev.npy', err_dev)

## Time to do Laplace Approximation.
print('MAP configuration reached: Calculating block diagonal Hessian.')

# Get hessian factors
EQ1, EHH1, MAP1, EQ2, EHH2, MAP2, EQ3, EHH3, MAP3 = net.get_K_laplace_params(trainloader)

h_params = [EQ1, EHH1, MAP1, EQ2, EHH2, MAP2, EQ3, EHH3, MAP3]
save_object(h_params, models_dir+'/block_hessian_params.pkl')
print('Hessian Parameters Saved')

data_scale = np.sqrt(len(trainset))
prior_sig = args.hessian_diag_sig
prior_prec = 1/prior_sig**2
prior_scale = np.sqrt(prior_prec)

# Scale and invert factors
# upper_Qinv, lower_HHinv
print('Scaling and inverting Hessian factors')
scale_inv_EQ1 = chol_scale_invert_kron_factor(EQ1, prior_scale, data_scale, upper=True)
scale_inv_EHH1 = chol_scale_invert_kron_factor(EHH1, prior_scale, data_scale, upper=False)

scale_inv_EQ2 = chol_scale_invert_kron_factor(EQ2, prior_scale, data_scale, upper=True)
scale_inv_EHH2 = chol_scale_invert_kron_factor(EHH2, prior_scale, data_scale, upper=False)

scale_inv_EQ3 = chol_scale_invert_kron_factor(EQ3, prior_scale, data_scale, upper=True)
scale_inv_EHH3 = chol_scale_invert_kron_factor(EHH3, prior_scale, data_scale, upper=False)



## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its

textsize = 15
marker = 5

plt.figure(dpi=100)
fig, ax1 = plt.subplots()
ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
ax1.plot(pred_cost_train, 'r--')
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
ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * err_dev[::nb_its_dev], 'b-')
ax2.semilogy(100 * err_train, 'r--')
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