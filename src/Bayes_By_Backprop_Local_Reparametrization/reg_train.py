from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid
from src.utils import *
from numpy.random import normal


def train_BNN_classification(net, name, batch_size, nb_epochs, trainset, valset, cuda,
                         burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                         re_burn, flat_ims=False, nb_its_dev=1):

    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    if cuda:
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
        err_train /= nb_samples
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

    return cost_train, cost_dev, err_train, err_dev


##########

def train_BNN_regression(net, name, batch_size, nb_epochs, trainset, valset, cuda,
                                 burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                                 re_burn, flat_ims=False, nb_its_dev=1, y_mu=1, y_std=1):

    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    if cuda:
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
    epoch = 0
    it_count = 0
    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    cost_train = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    ll_dev = np.zeros(nb_epochs)
    rms_dev = np.zeros(nb_epochs)
    best_cost = np.inf

    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        for x, y in trainloader:

            if flat_ims:
                x = x.view(x.shape[0], -1)

            cost_pred, _, _ = net.fit(x, y, burn_in=(i % re_burn < burn_in),
                                      resample_momentum=(it_count % resample_its == 0),
                                      resample_prior=(it_count % resample_prior_its == 0))
            it_count += 1
            cost_train[i] += cost_pred
            nb_samples += len(x)

        cost_train[i] /= nb_samples
        toc = time.time()

        # ---- print
        print("it %d/%d, Jtr_pred = %f, " % (i, nb_epochs, cost_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr(i)

        # ---- save weights
        if i % re_burn >= burn_in and i % sim_steps == 0:
            net.save_sampled_net(max_samples=N_saves)

        # ---- dev
        if i % nb_its_dev == 0:
            nb_samples = 0
            mu_vec = []
            sigma_vec = []
            y_vec = []
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                cost, mu, sigma = net.eval(x, y)
                mu_vec.append(mu.data.cpu())
                sigma_vec.append(sigma.data.cpu())
                y_vec.append(y.data.cpu())

                cost_dev[i] += cost
                nb_samples += len(x)

            mu_vec = torch.cat(mu_vec)
            sigma_vec = torch.cat(sigma_vec)
            y_vec = torch.cat(y_vec)
            rms, ll = net.unnormalised_eval(mu_vec, sigma_vec, y_vec, y_mu, y_std)
            ll_dev[i] = ll
            rms_dev[i] = rms
            cost_dev[i] /= nb_samples
            cprint('g', '    Jdev = %f \n' % cost_dev[i])
            cprint('g', ' Loglike = %.4f, rms = %.4f\n' % (ll, rms))
            if cost_dev[i] < best_cost:
                best_cost = cost_dev[i]
                cprint('b', 'best test error')

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)


    ## SAVE WEIGHTS
    net.save_weights(models_dir + '/state_dicts.pkl')

    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its
    textsize = 15
    marker = 5

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(cost_train, 'r--')
    ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('-loglike')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('train dev normalised ll')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, nb_epochs, nb_its_dev), ll_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('-loglike')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax = plt.gca()
    plt.title('un-normalised')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/ll.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, nb_epochs, nb_its_dev), rms_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('rms')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax = plt.gca()
    plt.title('un-normalised')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/rms.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return cost_train, cost_dev, rms_dev, ll_dev



