from src.priors import *
from src.base_net import *

import torch.nn.functional as F
import torch.nn as nn
import copy

def sample_weights(W_mu, b_mu, W_p, b_p):
    """Quick method for sampling weights and exporting weights"""
    eps_W = W_mu.data.new(W_mu.size()).normal_()
    # sample parameters
    std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
    W = W_mu + 1 * std_w * eps_W

    if b_mu is not None:
        std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
        eps_b = b_mu.data.new(b_mu.size()).normal_()
        b = b_mu + 1 * std_b * eps_b
    else:
        b = None

    return W, b


def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD


class BayesLinear_local_reparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """
    def __init__(self, n_in, n_out, prior_sig):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sig = prior_sig

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(
            torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

    def forward(self, X, sample=False):
        #         print(self.training)

        if not self.training and not sample:  # This is just a placeholder function
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        else:

            # calculate std
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
            act_W_std = torch.sqrt(torch.mm(X.pow(2), std_w.pow(2)))

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch output
            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
            eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1))

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            act_b_out = self.b_mu + std_b * eps_b

            output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)

            kld = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w) + KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu,
                                                                                      sig_q=std_b)
            return output, kld, 0


class bayes_linear_LR_2L(nn.Module):
    def __init__(self, input_dim, output_dim, nhid, prior_sig):
        super(bayes_linear_LR_2L, self).__init__()

        n_hid = nhid
        self.prior_sig = prior_sig

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bfc1 = BayesLinear_local_reparam(input_dim, n_hid, self.prior_sig)
        self.bfc2 = BayesLinear_local_reparam(n_hid, n_hid, self.prior_sig)
        self.bfc3 = BayesLinear_local_reparam(n_hid, output_dim, self.prior_sig)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=False):
        tlqw = 0
        tlpw = 0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, lqw, lpw = self.bfc1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = self.act(x)
        # -----------------
        x, lqw, lpw = self.bfc2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        # -----------------
        x = self.act(x)
        # -----------------
        y, lqw, lpw = self.bfc3(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw

        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

        return predictions, tlqw_vec, tlpw_vec


class BBP_Bayes_Net_LR(BaseNet):
    """Full network wrapper for Bayes By Backprop nets with methods for training, prediction and weight prunning"""
    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, batch_size=128, Nbatches=0,
                 nhid=1200, prior_sig=0.1):
        super(BBP_Bayes_Net_LR, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.classes = classes
        self.nhid = nhid
        self.prior_sig = prior_sig
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.side_in = side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = bayes_linear_LR_2L(input_dim=self.channels_in * self.side_in * self.side_in, output_dim=self.classes,
                                     nhid=self.nhid, prior_sig=self.prior_sig)
        if self.cuda:
            self.model = self.model.cuda()

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        self.optimizer.zero_grad()

        if samples == 1:
            out, tlqw, tlpw = self.model(x)
            mlpdw = F.cross_entropy(out, y, reduction='sum')
            Edkl = (tlqw - tlpw) / self.Nbatches

        elif samples > 1:
            mlpdw_cum = 0
            Edkl_cum = 0

            for i in range(samples):
                out, tlqw, tlpw = self.model(x, sample=True)
                mlpdw_i = F.cross_entropy(out, y, reduction='sum')
                Edkl_i = (tlqw - tlpw) / self.Nbatches
                mlpdw_cum = mlpdw_cum + mlpdw_i
                Edkl_cum = Edkl_cum + Edkl_i

            mlpdw = mlpdw_cum / samples
            Edkl = Edkl_cum / samples

        loss = Edkl + mlpdw
        loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return Edkl.data, mlpdw.data, err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self, Nsamples=10):
        state_dict = self.model.state_dict()
        weight_vec = []

        for i in range(Nsamples):
            previous_layer_name = ''
            for key in state_dict.keys():
                layer_name = key.split('.')[0]
                if layer_name != previous_layer_name:
                    previous_layer_name = layer_name

                    W_mu = state_dict[layer_name + '.W_mu'].data
                    W_p = state_dict[layer_name + '.W_p'].data

                    #                 b_mu = state_dict[layer_name+'.b_mu'].cpu().data
                    #                 b_p = state_dict[layer_name+'.b_p'].cpu().data

                    W, b = sample_weights(W_mu=W_mu, b_mu=None, W_p=W_p, b_p=None)

                    for weight in W.cpu().view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def get_weight_SNR(self, thresh=None):
        state_dict = self.model.state_dict()
        weight_SNR_vec = []

        if thresh is not None:
            mask_dict = {}

        previous_layer_name = ''
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name != previous_layer_name:
                previous_layer_name = layer_name

                W_mu = state_dict[layer_name + '.W_mu'].data
                W_p = state_dict[layer_name + '.W_p'].data
                sig_W = 1e-6 + F.softplus(W_p, beta=1, threshold=20)

                b_mu = state_dict[layer_name + '.b_mu'].data
                b_p = state_dict[layer_name + '.b_p'].data
                sig_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                W_snr = (torch.abs(W_mu) / sig_W)
                b_snr = (torch.abs(b_mu) / sig_b)

                if thresh is not None:
                    mask_dict[layer_name + '.W'] = W_snr > thresh
                    mask_dict[layer_name + '.b'] = b_snr > thresh

                else:

                    for weight_SNR in W_snr.cpu().view(-1):
                        weight_SNR_vec.append(weight_SNR)

                    for weight_SNR in b_snr.cpu().view(-1):
                        weight_SNR_vec.append(weight_SNR)

        if thresh is not None:
            return mask_dict
        else:
            return np.array(weight_SNR_vec)

    def get_weight_KLD(self, Nsamples=20, thresh=None):
        state_dict = self.model.state_dict()
        weight_KLD_vec = []

        if thresh is not None:
            mask_dict = {}

        previous_layer_name = ''
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name != previous_layer_name:
                previous_layer_name = layer_name

                W_mu = state_dict[layer_name + '.W_mu'].data
                W_p = state_dict[layer_name + '.W_p'].data
                b_mu = state_dict[layer_name + '.b_mu'].data
                b_p = state_dict[layer_name + '.b_p'].data

                std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                KL_W = W_mu.new(W_mu.size()).zero_()
                KL_b = b_mu.new(b_mu.size()).zero_()
                for i in range(Nsamples):
                    W, b = sample_weights(W_mu=W_mu, b_mu=b_mu, W_p=W_p, b_p=b_p)
                    # Note that this will currently not work with slab and spike prior
                    KL_W += isotropic_gauss_loglike(W, W_mu, std_w, do_sum=False) - self.model.prior_instance.loglike(W,
                                                                                                                      do_sum=False)
                    KL_b += isotropic_gauss_loglike(b, b_mu, std_b, do_sum=False) - self.model.prior_instance.loglike(b,
                                                                                                                      do_sum=False)

                KL_W /= Nsamples
                KL_b /= Nsamples

                if thresh is not None:
                    mask_dict[layer_name + '.W'] = KL_W > thresh
                    mask_dict[layer_name + '.b'] = KL_b > thresh

                else:

                    for weight_KLD in KL_W.cpu().view(-1):
                        weight_KLD_vec.append(weight_KLD)

                    for weight_KLD in KL_b.cpu().view(-1):
                        weight_KLD_vec.append(weight_KLD)

        if thresh is not None:
            return mask_dict
        else:
            return np.array(weight_KLD_vec)

    def mask_model(self, Nsamples=0, thresh=0):
        '''
        Nsamples is used to select SNR (0) or KLD (>0) based masking
        '''
        original_state_dict = copy.deepcopy(self.model.state_dict())
        state_dict = self.model.state_dict()

        if Nsamples == 0:
            mask_dict = self.get_weight_SNR(thresh=thresh)
        else:
            mask_dict = self.get_weight_KLD(Nsamples=Nsamples, thresh=thresh)

        n_unmasked = 0

        previous_layer_name = ''
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name != previous_layer_name:
                previous_layer_name = layer_name

                state_dict[layer_name + '.W_mu'][1 - mask_dict[layer_name + '.W']] = 0
                state_dict[layer_name + '.W_p'][1 - mask_dict[layer_name + '.W']] = -1000

                state_dict[layer_name + '.b_mu'][1 - mask_dict[layer_name + '.b']] = 0
                state_dict[layer_name + '.b_p'][1 - mask_dict[layer_name + '.b']] = -1000

                n_unmasked += mask_dict[layer_name + '.W'].sum()
                n_unmasked += mask_dict[layer_name + '.b'].sum()

        return original_state_dict, n_unmasked

