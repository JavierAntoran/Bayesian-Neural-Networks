from src.base_net import *
from hessian_operations import sample_K_laplace_MN, softmax_CE_preact_hessian, layer_act_hessian_recurse
import torch.nn as nn
import torch.nn.functional as F


class Linear_2L_KFRA(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L_KFRA, self).__init__()

        self.n_hid = n_hid

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, output_dim)

        # choose your non linearity
        self.act = nn.ReLU(inplace=True)

        self.one = None
        self.a2 = None
        self.h2 = None
        self.a1 = None
        self.h1 = None
        self.a0 = None

    def forward(self, x):
        self.one = x.new(x.shape[0], 1).fill_(1)

        a0 = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        self.a0 = torch.cat((a0.data, self.one), dim=1)
        # -----------------
        h1 = self.fc1(a0)
        self.h1 = h1.data  # torch.cat((h1, self.one), dim=1)
        # -----------------
        a1 = self.act(h1)
        #         a1.retain_grad()
        self.a1 = torch.cat((a1.data, self.one), dim=1)
        # -----------------
        h2 = self.fc2(a1)
        self.h2 = h2.data  # torch.cat((h2, self.one), dim=1)
        # -----------------
        a2 = self.act(h2)
        #         a2.retain_grad()
        self.a2 = torch.cat((a2.data, self.one), dim=1)
        # -----------------
        h3 = self.fc3(a2)

        return h3

    def sample_predict(self, x, Nsamples, Qinv1, HHinv1, MAP1, Qinv2, HHinv2, MAP2, Qinv3, HHinv3, MAP3):
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        x = x.view(-1, self.input_dim)
        for i in range(Nsamples):
            # -----------------
            w1, b1 = sample_K_laplace_MN(MAP1, Qinv1, HHinv1)
            a = torch.matmul(x, torch.t(w1)) + b1.unsqueeze(0)
            a = self.act(a)
            # -----------------
            w2, b2 = sample_K_laplace_MN(MAP2, Qinv2, HHinv2)
            a = torch.matmul(a, torch.t(w2)) + b2.unsqueeze(0)
            a = self.act(a)
            # -----------------
            w3, b3 = sample_K_laplace_MN(MAP3, Qinv3, HHinv3)
            y = torch.matmul(a, torch.t(w3)) + b3.unsqueeze(0)
            predictions[i] = y

        return predictions


class KBayes_Net(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, n_hid=1200, batch_size=128, prior_sig=0):
        super(KBayes_Net, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.n_hid = n_hid
        self.prior_sig = prior_sig
        self.classes = classes
        self.batch_size = batch_size
        self.side_in = side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = Linear_2L_KFRA(input_dim=self.channels_in * self.side_in * self.side_in, output_dim=self.classes,
                                    n_hid=self.n_hid)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5,
                                         weight_decay=(1 / self.prior_sig ** 2))

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def fit(self, x, y):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='sum')

        loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def get_K_laplace_params(self, trainloader):
        """Do pass through full training set in order to get the expected layer-wise hessian product factors.
        This is done by calculating the hessian for the first layer's activations and propagating them backward
        recursively."""
        self.model.eval()

        it_counter = 0
        cum_HH1 = self.model.fc1.weight.data.new(self.model.n_hid, self.model.n_hid).fill_(0)
        cum_HH2 = self.model.fc1.weight.data.new(self.model.n_hid, self.model.n_hid).fill_(0)
        cum_HH3 = self.model.fc1.weight.data.new(self.model.output_dim, self.model.output_dim).fill_(0)

        cum_Q1 = self.model.fc1.weight.data.new(self.model.input_dim + 1, self.model.input_dim + 1).fill_(0)
        cum_Q2 = self.model.fc1.weight.data.new(self.model.n_hid + 1, self.model.n_hid + 1).fill_(0)
        cum_Q3 = self.model.fc1.weight.data.new(self.model.n_hid + 1, self.model.n_hid + 1).fill_(0)

        # Forward pass

        for x, y in trainloader:
            x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

            self.optimizer.zero_grad()

            out = self.model(x)
            out_act = F.softmax(out, dim=1)
            loss = F.cross_entropy(out, y, reduction='sum')

            loss.backward()

            #     ------------------------------------------------------------------
            HH3 = softmax_CE_preact_hessian(out_act.data)
            cum_HH3 += HH3.sum(dim=0)
            #     print(model.a2.data.shape)
            Q3 = torch.bmm(self.model.a2.data.unsqueeze(2), self.model.a2.data.unsqueeze(1))
            cum_Q3 += Q3.sum(dim=0)
            #     ------------------------------------------------------------------
            HH2 = layer_act_hessian_recurse(prev_hessian=HH3, prev_weights=self.model.fc3.weight.data,
                                            layer_pre_acts=self.model.h2.data)
            cum_HH2 += HH2.sum(dim=0)
            Q2 = torch.bmm(self.model.a1.data.unsqueeze(2), self.model.a1.data.unsqueeze(1))
            cum_Q2 += Q2.sum(dim=0)
            #     ------------------------------------------------------------------
            HH1 = layer_act_hessian_recurse(prev_hessian=HH2, prev_weights=self.model.fc2.weight.data,
                                            layer_pre_acts=self.model.h1.data)
            cum_HH1 += HH1.sum(dim=0)
            Q1 = torch.bmm(self.model.a0.data.unsqueeze(2), self.model.a0.data.unsqueeze(1))
            cum_Q1 += Q1.sum(dim=0)
            #     ------------------------------------------------------------------
            it_counter += x.shape[0]
            # print(it_counter)

        EHH3 = cum_HH3 / it_counter
        EHH2 = cum_HH2 / it_counter
        EHH1 = cum_HH1 / it_counter

        EQ3 = cum_Q3 / it_counter
        EQ2 = cum_Q2 / it_counter
        EQ1 = cum_Q1 / it_counter

        MAP3 = torch.cat((self.model.fc3.weight.data, self.model.fc3.bias.data.unsqueeze(1)), dim=1)
        MAP2 = torch.cat((self.model.fc2.weight.data, self.model.fc2.bias.data.unsqueeze(1)), dim=1)
        MAP1 = torch.cat((self.model.fc1.weight.data, self.model.fc1.bias.data.unsqueeze(1)), dim=1)

        return EQ1, EHH1, MAP1, EQ2, EHH2, MAP2, EQ3, EHH3, MAP3

    def sample_eval(self, x, y, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2, MAP2,
                    scale_inv_EQ3, scale_inv_EHH3, MAP3, logits=False):
        """Prediction, only returining result with weights marginalised"""
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2,
                                        MAP2, scale_inv_EQ3, scale_inv_EHH3, MAP3)

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

    def all_sample_eval(self, x, y, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2, MAP2,
                        scale_inv_EQ3, scale_inv_EHH3, MAP3):
        """Returns predictions for each MC sample"""
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model.sample_predict(x, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2,
                                        MAP2, scale_inv_EQ3, scale_inv_EHH3, MAP3)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self, Nsamples, scale_inv_EQ1, scale_inv_EHH1, MAP1, scale_inv_EQ2, scale_inv_EHH2, MAP2,
                           scale_inv_EQ3, scale_inv_EHH3, MAP3):
        weight_vec = []

        for i in range(Nsamples):

            w1, b1 = sample_K_laplace_MN(MAP1, scale_inv_EQ1, scale_inv_EHH1)
            w2, b2 = sample_K_laplace_MN(MAP2, scale_inv_EQ2, scale_inv_EHH2)
            w3, b3 = sample_K_laplace_MN(MAP3, scale_inv_EQ3, scale_inv_EHH3)

            for weight in w1.cpu().numpy().flatten():
                weight_vec.append(weight)
            for weight in w2.cpu().numpy().flatten():
                weight_vec.append(weight)
            for weight in w3.cpu().numpy().flatten():
                weight_vec.append(weight)

        return np.array(weight_vec)