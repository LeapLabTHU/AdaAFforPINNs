from curses.panel import new_panel
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from choose_optimizer import *
from matplotlib import pyplot as plt 
import os 
from utils import *
import copy 
from mpl_toolkits.axes_grid1 import make_axes_locatable

import math 
import random 
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm 

all_features = []
enable_retain_grad = False 
all_grad = []


class Rational(nn.Module):

    def __init__(self, width=1):
        super().__init__()
        w_numerator = [
            2.1172949817857366e-09,
            0.9999942495075363,
            6.276332768876106e-07,
            0.10770864506559906,
            2.946556898117109e-08,
            0.000871124373591946
        ]
        w_denominator = [
            6.376908337817277e-07,
            0.44101418051922986,
            2.2747661404467182e-07,
            0.014581039909092108
        ]

        # self.numerator = nn.Parameter(torch.tensor(w_numerator).double(),requires_grad=True)
        # self.denominator = nn.Parameter(torch.tensor(w_denominator).double(),requires_grad=True)
        self.numerator = nn.Parameter(torch.randn(6).double(),requires_grad=True)
        self.denominator = nn.Parameter(torch.randn(4).double(),requires_grad=True)

    def _get_xps(self, z, len_numerator, len_denominator):
        xps = list()
        xps.append(z)
        for _ in range(max(len_numerator, len_denominator) - 2):
            xps.append(xps[-1].mul(z))
        xps.insert(0, torch.ones_like(z))
        return torch.stack(xps, 1)

    def forward(self, x):
      
        numerator = sum([self.numerator[order] * torch.pow(x, order) for order in range(self.numerator.shape[0])])
        denominator = sum([self.denominator[order] * torch.pow(x, order+1) for order in range(self.denominator.shape[0])])
        return numerator.div(1 + denominator.abs())

class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width=1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width).double())
        self.p2 = nn.Parameter(torch.randn(1, width).double())
        self.beta = nn.Parameter(torch.ones(1, width).double())

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


class Sine(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(Sine, self).__init__()
        self.alpha = alpha
        print(self.alpha)
    def forward(self, x):
        return torch.sin(self.alpha * x)

class Cosine(torch.nn.Module):
    def __init__(self,alpha=1.0):
        super(Cosine, self).__init__()
        self.alpha = alpha
        print(self.alpha)
    def forward(self, x):
        return torch.cos(self.alpha * x)

class Exp(torch.nn.Module):
    def __init__(self, alpha=2.0):
        super(Exp, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.exp((x-0.0)/self.alpha)-1.0

class Pow(torch.nn.Module):
    def __init__(self, order, scale=False):
        super(Pow, self).__init__()
        self.order = order
        self.scale = scale

    def forward(self, x):
        # print(self.alpha)
        # return torch.sign(x) * (torch.exp(-torch.abs(x)) - 1.0)
        if self.scale and self.order > 0:
            return torch.pow(x, self.order) / (1.0 * self.order)
        else:
            return torch.pow(x, self.order)

class Log(torch.nn.Module):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        return torch.log(torch.abs(x)+1e-8)

class MyLinearPool(torch.nn.Module):
    def __init__(self, input_channel, output_channel, poolsize='0', aggregate='sum', weight_sharing=True, out_list=True, channel_wise=True, use_norm=False, exp_alpha=2.0, sin_alpha=1.0, tau=1.0, scaler=1.0, enable_scaling=False, random_init=False, taylor=False, taylor_scale=False, taylor_order=5):
        # print(exp_alpha)
        super(MyLinearPool, self).__init__()
        
        act_pool = [Sine(alpha=sin_alpha), torch.nn.Tanh(), nn.GELU(), nn.SiLU(), nn.Softplus()]
        self.poolsize = [int(i) for i in poolsize.split(',')]
        if taylor:
            self.activations = torch.nn.ModuleList([Pow(order=order_i, scale=taylor_scale) for order_i in range(taylor_order+1)])
        else:
            self.activations = torch.nn.ModuleList([act_pool[i] for i in self.poolsize])
        self.aggregate = aggregate
        self.tau = tau 
        print('tau', self.tau)

        self.use_norm = use_norm
        if use_norm:
            self.norm = torch.nn.LayerNorm(output_channel,elementwise_affine=False).double()
        self.enable_scaling = enable_scaling
        self.scaler = scaler 
        self.weight_sharing = weight_sharing
        self.enable_scaling = enable_scaling
        self.scaler = scaler 
        if self.weight_sharing:
            self.fc = torch.nn.Linear(input_channel, output_channel).double()
        else:
            self.fc = torch.nn.ModuleList([torch.nn.Linear(input_channel, output_channel).double() for _ in range(len(self.activations))])
        if channel_wise:
            if self.aggregate == 'sigmoid' or self.aggregate == 'softmax':
                if random_init:
                    coeff_weight = torch.randn(len(self.activations), output_channel).double()
                    self.coeff = torch.nn.parameter.Parameter(coeff_weight)
                else:
                    self.coeff = torch.nn.parameter.Parameter(torch.zeros(len(self.activations), output_channel).double())
            else:
               
                self.coeff = torch.nn.parameter.Parameter(torch.ones(len(self.activations), output_channel).double() / len(self.poolsize))
       
            self.coeff2 = torch.nn.parameter.Parameter(1.0 / scaler * torch.ones(len(self.activations), output_channel).double())
        else:
            if self.aggregate == 'sigmoid' or self.aggregate == 'softmax':
                if random_init:
                    coeff_weight = torch.randn(len(self.activations), 1).double()
                    self.coeff = torch.nn.parameter.Parameter(coeff_weight)
                else:
                    self.coeff = torch.nn.parameter.Parameter(torch.zeros(len(self.activations), 1).double())
            else:
               
                self.coeff = torch.nn.parameter.Parameter(torch.randn(len(self.activations), 1).double() / len(self.poolsize))
            self.coeff2 = torch.nn.parameter.Parameter(1.0 / scaler * torch.ones(len(self.activations), 1).double())
        self.out_list = out_list
        
    def forward(self, input):
        if self.aggregate == 'sigmoid':
            coeff = torch.sigmoid(self.coeff)
        elif self.aggregate == 'softmax':
            coeff = torch.softmax(self.coeff/self.tau, dim=0)
        elif self.aggregate == 'unlimited':
            coeff = self.coeff 
        elif self.aggregate == 'l1_norm':
            coeff = self.coeff / self.coeff.abs().sum(dim=0)
        else:
            raise NotImplementedError

        if self.enable_scaling:
            coeff2 = self.coeff2
        else:
            coeff2 = self.coeff2.detach()

        x = input[0]
        detach = input[1]
        # print(coeff.shape)
        if detach:
            if self.weight_sharing:
                y = self.fc(x) #* self.coeff2
                out = [c.detach()*act(self.scaler * c2 * y) for act, c, c2 in zip(self.activations, coeff, coeff2)]
            else:
                out = [c.detach()*act(self.scaler * c2 * layer(x)) for act, layer, c, c2 in zip(self.activations, self.fc, coeff, coeff2)]
        else:
            if self.weight_sharing:
                y = self.fc(x)
                if self.use_norm:
                    y = self.norm(y)
                # print(y.mean())
                out = [c*act(self.scaler * c2 * y) for act, c, c2 in zip(self.activations, coeff, coeff2)]
                # out_mean = [o.mean() for o in out]
                # print(out_mean)
            else:
                out = [c*act(self.scaler * c2 * layer(x)) for act, layer, c, c2 in zip(self.activations, self.fc, coeff, coeff2)]
    
        if self.out_list:
            return [sum(out), detach]
        else:
            return sum(out)

class LLAF(torch.nn.Module):
    def __init__(self, input_channel, output_channel, scaler=10, channel_wise=False):
        super(LLAF, self).__init__()
     
        self.fc = torch.nn.Linear(input_channel, output_channel).double()
        self.channel_wise = channel_wise 
        if self.channel_wise:
            self.coeff2 = torch.nn.parameter.Parameter(1.0 / scaler * torch.ones(output_channel).double())
        else:
            self.coeff2 = torch.nn.parameter.Parameter(1.0 / scaler * torch.ones(1).double())
        self.scaler = scaler 
        self.disable = False 
        print(self.scaler)
    def forward(self, x):
        if self.disable:
            y = self.fc(x) * self.coeff2.detach() * self.scaler 
        else:
            y = self.fc(x) * self.coeff2 * self.scaler 
        return y 

class DNN(torch.nn.Module):
    def __init__(self, args, layers, activation, linearpool=False, init=False, poolsize=1, aggregate='sum', llaf=False):
        super(DNN, self).__init__()
        self.args = args 

        # parameters
        self.depth = len(layers) - 1
        self.aggregate = aggregate 
        if self.aggregate == 'cat':
            for i in range(1, len(layers)-1):
                layers[i] = layers[i] * poolsize
        # activations = activations.split(',')
        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'softplus':
            self.activation = torch.nn.Softplus
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        elif activation == 'cos':
            self.activation = Cosine
        elif activation == 'exp':
            self.activation = Exp
        elif activation == 'logsigmoid':
            self.activation = torch.nn.LogSigmoid
        elif activation == 'silu':
            self.activation = torch.nn.SiLU
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid
        elif activation == 'elu':
            self.activation = torch.nn.ELU
        elif activation == 'aconc':
            self.activation = AconC
        elif activation == 'rational':
            self.activation = Rational
        else:
            raise NotImplementedError

        self.detach = False

        layer_list = list()
      
        if linearpool:
            for i in range(self.depth-1):
                out_list = False if i == self.depth-2 else True
                layer_list.append(
                        ('layer_%d' % i, MyLinearPool(layers[i], layers[i+1], poolsize=poolsize, aggregate=aggregate, weight_sharing=args.weight_sharing, out_list=out_list, channel_wise=self.args.channel_wise, use_norm=self.args.use_norm, exp_alpha=self.args.exp_alpha, sin_alpha=self.args.sin_alpha, tau=self.args.tau, scaler=self.args.scaler, enable_scaling=self.args.enable_scaling, random_init=self.args.random_init, taylor=args.taylor, taylor_order=args.taylor_order, taylor_scale=args.taylor_scale))
                    )
            layer_list.append(
                    ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]).double())
                )
        else:
            for i in range(self.depth - 1):
                if llaf:
                    layer_list.append(
                        ('layer_%d' % i, LLAF(layers[i], layers[i+1], self.args.scaler, self.args.channel_wise))
                    )
                else:
                    layer_list.append(
                        ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]).double())
                    )
                 
                if activation == 'exp':
                    layer_list.append(('activation_%d' % i, self.activation(alpha=self.args.exp_alpha)))
                elif activation == 'sin':
                    layer_list.append(('activation_%d' % i, self.activation(alpha=self.args.sin_alpha)))
                
                # elif activation == 'rational':
                #     layer_list.append(('activation_%d' % i, self.activation().double()))
                else:
                    layer_list.append(('activation_%d' % i, self.activation()))
            
           
            layer_list.append(
                ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]).double())
            )
        
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        if init:
            self.init_weights()
    
    def init_weights(self):
        for name,m in self.layers.named_modules():
            print(name)
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                print('init_xavier in backbone')
                if 'layer_0' in name:
                    gain = self.args.gain_0
                else:
                    gain = self.args.gain
                nn.init.xavier_uniform_(m.weight, gain)
                if 'layer_0' in name:
                    if self.args.enable_vx:
                        with torch.no_grad():
                            # import pdb 
                            # pdb.set_trace()
                            m.weight[:,0] =  m.weight[:,0] / 8.0
                            # m.weight[:,1] =  m.weight[:,1] * 8.0
                    if self.args.enable_vt:
                        with torch.no_grad():
                            # m.weight[:,0] =  m.weight[:,0] / 8.0
                            m.weight[:,1] =  m.weight[:,1] * 8.0
                        
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.args.linearpool:
            out = self.layers([x, self.detach])
        else:
            out = self.layers(x)
        return out


class PhysicsInformedNN_pbc():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, args, repeat, logger, device, system, X_star_noinitial_noboundary, X_u_train, u_train, X_f_train, u22, bc_lb, bc_ub, X_star, Exact, layers, G, nu, beta, rho, optimizer_name, lr,
        net, activation='tanh'):
        self.args = args 
        self.repeat = repeat
        self.logger = logger 
        self.system = system
        self.device = device 
        self.X_star_noinitial_noboundary = X_star_noinitial_noboundary
        self.X_star = X_star 
        self.Exact = Exact if isinstance(Exact, list) else [Exact]

        # print(device)
        # a = torch.randn(2).cuda()
        # print(a)
        self.x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).double().to(device)
        # print(self.x_u)
        self.t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).double().to(device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).double().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).double().to(device)
        # import pdb 
        # pdb.set_trace()
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).double().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).double().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).double().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).double().to(device)
     
        self.net = net

        self.use_auxiliary = args.use_auxiliary

        self.depth = len(layers) - 1

        self.num_head = args.num_head
        self.writer = SummaryWriter(log_dir=os.path.join(args.work_dir, f'run_{repeat}')) 

        
        self.u = torch.tensor(u_train, requires_grad=False).double().to(device)
        # if self.system == 'CH' and self.args.decouple:
        #     self.u22 = torch.tensor(u22, requires_grad=False).double().to(device)
        #     self.u = torch.cat([self.u, self.u22], dim=-1)

        self.layers = layers
        self.nu = nu if isinstance(nu, list) else [nu]
        self.beta = beta if isinstance(beta, list) else [beta]
        self.rho = rho if isinstance(rho, list) else [rho]
        assert len(self.nu) == args.num_head 
        assert len(self.beta) == args.num_head 
        assert len(self.rho) == args.num_head 
        assert len(self.Exact) == args.num_head 
        print(self.nu)
        print(self.beta)
        print(self.rho)
        self.G = torch.tensor(G, requires_grad=True).double().to(device)
        self.G = self.G.reshape(-1, 1)

        self.L_f = args.L_f 
        self.L_u = args.L_u 
        self.L_b = args.L_b

        self.lr = lr
        self.optimizer_name = optimizer_name
            
        self.dnn = DNN(args, layers, activation, linearpool=args.linearpool, init=args.init, poolsize=args.poolsize, aggregate=args.aggregate, llaf=args.llaf).to(device)
              
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location='cpu')
            self.logger.info(f'load checkpoint {args.resume}')
            # new_state_dict = {}
            # for k,v in checkpoint['state_dict'].items():
            #     if 'coeff2' in k:
            #         new_state_dict[k] = v.new_ones((5,1))
            #     else:
            #         new_state_dict[k] = v
            # self.dnn.load_state_dict(new_state_dict, strict=False)
            
            
            self.dnn.load_state_dict(checkpoint['state_dict'], strict=False)
   
        self.dnn_auxiliary = copy.deepcopy(self.dnn)
      
        self.logger.info(self.dnn)
   
        self.iter = 0
    

    def net_u(self, x, t, use_auxiliary=False, detach=False):
        self.dnn.detach = detach
        """The standard DNN that takes (x,t) --> u."""
       
        input_xt = torch.cat([x, t], dim=1)
          
        if use_auxiliary:
            u = self.dnn_auxiliary(input_xt)
        else:
            u = self.dnn(input_xt)
        if 'AC' in self.system and self.args.hard_ibc:
            u = x**2 * torch.cos(np.pi * x) + t * (1 - x**2) * u
        if 'convection' in self.system and self.args.hard_ibc:
            u = t * u + torch.sin(x)
        if 'KdV' in self.system and self.args.hard_ibc:
            u = torch.cos(np.pi * x) + t * u
        if 'CH' in self.system and self.args.hard_ibc:
            if self.args.decouple:
                u_init = torch.cos(np.pi * x) - torch.exp(-4* (np.pi * x)**2)
                u2_init = - (np.pi ** 2) * torch.cos(np.pi * x) + 8* np.pi**2 * (1-8*(np.pi*x)**2) * torch.exp(-4* (np.pi * x)**2)
                y_init = torch.cat([u_init,u2_init],dim=-1)
                u = y_init + t * u
            else:
                u = torch.cos(np.pi * x) - torch.exp(-4* (np.pi * x)**2) + t * u
         
        if not isinstance(u, list):
            u = [u]
        return u

    def net_f(self, x, t, use_auxiliary=False, return_gradient=False, detach=False):
        """ Autograd for calculating the residual for different systems."""
        u = self.net_u(x, t, use_auxiliary=use_auxiliary, detach=detach)
        f_all = []
        u_tx = []
        for output_i, nu, beta, rho in zip(u, self.nu, self.beta, self.rho):
            if self.system == 'CH' and self.args.decouple:
                u_i = output_i[:, 0:1]
                u2_i = output_i[:, 1:2]
            else:
                u_i = output_i
            u_t = torch.autograd.grad(
                u_i, t,
                grad_outputs=torch.ones_like(u_i),
                retain_graph=True,
                create_graph=True
            )[0]
            u_x = torch.autograd.grad(
                u_i, x,
                grad_outputs=torch.ones_like(u_i),
                retain_graph=True,
                create_graph=True
            )[0]
            if 'inviscid' not in self.system:
                u_xx = torch.autograd.grad(
                    u_x, x,
                    grad_outputs=torch.ones_like(u_x),
                    retain_graph=True,
                    create_graph=True
                    )[0]
            u_tx.append([u_t.detach(), u_x.detach()])
            if 'convection' in self.system or 'diffusion' in self.system:
                f = u_t - nu*u_xx + beta*u_x - self.G
            elif 'rd' in self.system:
                f = u_t - nu*u_xx - rho*u_i + rho*u_i**2
            elif 'reaction' in self.system:
                f = u_t - rho*u_i + rho*u_i**2
            elif 'burger' in self.system:
                f = u_t + u_i * u_x - 0.01 / math.pi * u_xx 
            elif 'inviscid' in self.system:
                f = u_t + 2 * u_i * u_x 
            elif 'AC' in self.system:
                f = u_t - 0.001 * u_xx - 5 * (u_i - u_i**3)
                # f = u_t - u_xx
            elif 'KdV' in self.system:
                u_xxx = torch.autograd.grad(
                    u_xx, x,
                    grad_outputs=torch.ones_like(u_xx),
                    retain_graph=True,
                    create_graph=True
                )[0]
                f = u_t + u_i * u_x + 0.0025 * u_xxx 
            elif 'CH' in self.system:
                if self.args.decouple:
                    
                    y_ch = -0.02 * u2_i + u_i**3 - u_i
                    y_ch_x = torch.autograd.grad(
                        y_ch, x,
                        grad_outputs=torch.ones_like(y_ch),
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    y_ch_xx = torch.autograd.grad(
                        y_ch_x, x,
                        grad_outputs=torch.ones_like(y_ch_x),
                        retain_graph=True,
                        create_graph=True
                        )[0]
                    f1 = u_t - y_ch_xx
                    f2 = u2_i - u_xx
                    f = torch.cat([f1,f2], dim=-1)
                else:
                    y_ch = u_i**3 - u_i - 0.02 * u_xx 
                    y_ch_x = torch.autograd.grad(
                        y_ch, x,
                        grad_outputs=torch.ones_like(y_ch),
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    y_ch_xx = torch.autograd.grad(
                        y_ch_x, x,
                        grad_outputs=torch.ones_like(y_ch_x),
                        retain_graph=True,
                        create_graph=True
                        )[0]
                    f = u_t - y_ch_xx
            else:
                raise NotImplementedError
           
            f_all.append(f)
        if return_gradient:
            return f_all, u_tx
        else:
            return f_all

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x

    def net_b_derivatives_high_order(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_lb_xx = torch.autograd.grad(
            u_lb_x, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb_x),
            retain_graph=True,
            create_graph=True
            )[0]

        u_lb_xxx = torch.autograd.grad(
            u_lb_xx, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb_xx),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]
        
        u_ub_xx = torch.autograd.grad(
            u_ub_x, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub_x),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_xxx = torch.autograd.grad(
            u_ub_xx, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub_xx),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x, u_lb_xx, u_ub_xx, u_lb_xxx, u_ub_xxx

        
    def adapt_sample_range(self):
        t_range = np.linspace(0, 1, self.args.sample_stage+1)
        stage_iter = self.args.epoch // self.args.sample_stage
        if self.iter >= self.args.epoch:
            X_sample = self.X_star_noinitial_noboundary
        else:
            t_range_iter = t_range[self.iter // stage_iter+1]
            X_sample = self.X_star_noinitial_noboundary[self.X_star_noinitial_noboundary[:, 1] <= t_range_iter]
            # import pdb 
            # pdb.set_trace()
        return X_sample

    def loss_pinn(self, verbose=False, step=False, evaluate=False):
        """ Loss function. """
        
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            if self.args.sep_optim:
                self.optimizer_coeff.zero_grad()
        
        u_pred = self.net_u(self.x_u, self.t_u, detach=self.args.detach_u)
        
        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb, detach=self.args.detach_b)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub, detach=self.args.detach_b)

        if 'CH' in self.system and self.args.decouple:
            if not self.args.four_order:
                u_pred_lb_x, u_pred_ub_x = list(), list() 
                # u2_pred_lb_x, u2_pred_ub_x = list(), list() 
                for u_pred_lb_i, u_pred_ub_i in zip(u_pred_lb, u_pred_ub):
                    
                    u_pred_lb_x_i, u_pred_ub_x_i = self.net_b_derivatives(u_pred_lb_i[:, 0:1], u_pred_ub_i[:, 0:1], self.x_bc_lb, self.x_bc_ub)
                    u_pred_lb_x.append(u_pred_lb_x_i)
                    u_pred_ub_x.append(u_pred_ub_x_i)
            else:
                # u_pred_lb_x, u_pred_ub_x, u_pred_lb_xx, u_pred_ub_xx, u_pred_lb_xxx, u_pred_ub_xxx = list(), list(), list(), list(), list(), list()
                u_pred_lb_x, u_pred_ub_x = list(), list()
                u2_pred_lb_x, u2_pred_ub_x = list(), list() 
                for u_pred_lb_i, u_pred_ub_i in zip(u_pred_lb, u_pred_ub):
                    
                    # u_pred_lb_x_i, u_pred_ub_x_i, u_pred_lb_xx_i, u_pred_ub_xx_i, u_pred_lb_xxx_i, u_pred_ub_xxx_i = self.net_b_derivatives_high_order(u_pred_lb_i[:, 0:1], u_pred_ub_i[:, 0:1], self.x_bc_lb, self.x_bc_ub)
                    # u_pred_lb_x.append(u_pred_lb_x_i)
                    # u_pred_ub_x.append(u_pred_ub_x_i)
                    # u_pred_lb_xx.append(u_pred_lb_xx_i)
                    # u_pred_ub_xx.append(u_pred_ub_xx_i)
                    # u_pred_lb_xxx.append(u_pred_lb_xxx_i)
                    # u_pred_ub_xxx.append(u_pred_ub_xxx_i)
                    u_pred_lb_x_i, u_pred_ub_x_i = self.net_b_derivatives(u_pred_lb_i[:, 0:1], u_pred_ub_i[:, 0:1], self.x_bc_lb, self.x_bc_ub)
                    u2_pred_lb_x_i, u2_pred_ub_x_i = self.net_b_derivatives(u_pred_lb_i[:, 1:2], u_pred_ub_i[:, 1:2], self.x_bc_lb, self.x_bc_ub)
                    u2_pred_lb_x.append(u2_pred_lb_x_i)
                    u2_pred_ub_x.append(u2_pred_ub_x_i)
                    u_pred_lb_x.append(u_pred_lb_x_i)
                    u_pred_ub_x.append(u_pred_ub_x_i)
        else:
            u_pred_lb_x, u_pred_ub_x = list(), list() 
            for nu, u_pred_lb_i, u_pred_ub_i in zip(self.nu, u_pred_lb, u_pred_ub):
                if nu != 0 or 'KdV' in self.system or self.args.high_bc:
                    u_pred_lb_x_i, u_pred_ub_x_i = self.net_b_derivatives(u_pred_lb_i, u_pred_ub_i, self.x_bc_lb, self.x_bc_ub)
                    u_pred_lb_x.append(u_pred_lb_x_i)
                    u_pred_ub_x.append(u_pred_ub_x_i)

        f_pred = self.net_f(self.x_f, self.t_f, detach=self.args.detach_f)
        loss_u_list, loss_b_list, loss_f_list = list(), list(), list()
        # loss_u_list, loss_b_list = list(), list()
        # import pdb; pdb.set_trace()
        for idx in range(self.num_head):
                
            if 'CH' in self.system and self.args.decouple:
                loss_u_i = torch.mean((self.u - u_pred[idx][:,0:1]) ** 2)
            else:
                loss_u_i = torch.mean((self.u - u_pred[idx]) ** 2)

            loss_f_i = torch.mean(f_pred[idx] ** 2)
            
            if 'burger' in self.system or 'inviscid' in self.system:
                loss_b_i = torch.mean(u_pred_lb[idx] ** 2) + torch.mean(u_pred_ub[idx] ** 2)
        
            elif 'AC' in self.system:
                loss_b_i = torch.mean((u_pred_lb[idx]+1) ** 2) + torch.mean((u_pred_ub[idx]+1) ** 2)
            elif 'KdV' in self.system:
                loss_b_i = torch.mean((u_pred_lb[idx] - u_pred_ub[idx]) ** 2) + torch.mean((u_pred_lb_x[idx] - u_pred_ub_x[idx]) ** 2)
                # loss_b_i = 0
            elif 'CH' in self.system and self.args.decouple:
                # loss_b_i = 2 * torch.mean((u_pred_lb[idx] - u_pred_ub[idx]) ** 2) + torch.mean((u_pred_lb_x[idx] - u_pred_ub_x[idx]) ** 2) + torch.mean((u2_pred_lb_x[idx] - u2_pred_ub_x[idx]) ** 2)
                if self.args.four_order:
                #     loss_b_i = torch.mean((u_pred_lb[idx][:,0:1] - u_pred_ub[idx][:,0:1]) ** 2) + torch.mean((u_pred_lb_x[idx] - u_pred_ub_x[idx]) ** 2) \
                #         + torch.mean((u_pred_lb_xx[idx] - u_pred_ub_xx[idx]) ** 2) + torch.mean((u_pred_lb_xxx[idx] - u_pred_ub_xxx[idx]) ** 2)
                    loss_b_i = 2 * torch.mean((u_pred_lb[idx] - u_pred_ub[idx]) ** 2) + torch.mean((u_pred_lb_x[idx] - u_pred_ub_x[idx]) ** 2) + torch.mean((u2_pred_lb_x[idx] - u2_pred_ub_x[idx]) ** 2)
                else:
                    loss_b_i = torch.mean((u_pred_lb[idx][:,0:1] - u_pred_ub[idx][:,0:1]) ** 2) #+ torch.mean((u_pred_lb_x[idx] - u_pred_ub_x[idx]) ** 2)
            else:
                loss_b_i = torch.mean((u_pred_lb[idx] - u_pred_ub[idx]) ** 2)
                
            if self.nu[idx] != 0 or self.args.high_bc:
                loss_b_i += torch.mean((u_pred_lb_x[idx] - u_pred_ub_x[idx]) ** 2)
                
            loss_u_list.append(loss_u_i)
            loss_f_list.append(loss_f_i)
            loss_b_list.append(loss_b_i)
        loss_u = sum(loss_u_list) / self.num_head
        loss_f = sum(loss_f_list) / self.num_head
        loss_b = sum(loss_b_list) / self.num_head

        for name, p in self.dnn.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if verbose:
                        if (self.iter < self.args.epoch and self.iter % 100== 0) or (self.iter >= self.args.epoch and self.iter % self.args.print_freq == 0):
                            if 'coeff2' not in name and 'bias' not in name:
                                grad_u = torch.autograd.grad(
                                    loss_u, p,
                                    grad_outputs=torch.ones_like(loss_u),
                                    retain_graph=True,
                                    create_graph=False,
                                    allow_unused=True, 
                                    )[0]
                                self.writer.add_scalars(f'{name}_grad', {'grad_u_mean':grad_u.detach().abs().mean().item()}, self.iter)
                                self.writer.add_scalars(f'{name}_grad', {'grad_u_max': grad_u.detach().abs().max().item()}, self.iter)
                                grad_b = torch.autograd.grad(
                                                loss_b, p,
                                                grad_outputs=torch.ones_like(loss_b),
                                                retain_graph=True,
                                                create_graph=False,
                                                allow_unused=True, 
                                                )[0]
                                self.writer.add_scalars(f'{name}_grad', {'grad_b_mean':grad_b.detach().abs().mean().item()}, self.iter)
                                self.writer.add_scalars(f'{name}_grad', {'grad_b_max': grad_b.detach().abs().max().item()}, self.iter)
                                grad_f = torch.autograd.grad(
                                                loss_f, p,
                                                grad_outputs=torch.ones_like(loss_f),
                                                retain_graph=True,
                                                create_graph=False,
                                                allow_unused=True, 
                                                )[0]
                                self.writer.add_scalars(f'{name}_grad', {'grad_f_mean':grad_f.detach().abs().mean().item()}, self.iter)
                                self.writer.add_scalars(f'{name}_grad', {'grad_f_max': grad_f.detach().abs().max().item()}, self.iter)


        loss = self.L_u*loss_u + self.L_b*loss_b + self.L_f*loss_f
        # loss = self.L_u*loss_u + self.L_b*loss_b
        if evaluate:
            return loss.detach().cpu().numpy(), loss_u.detach().cpu().numpy(), loss_b.detach().cpu().numpy(), loss_f.detach().cpu().numpy() 

        # for l-bfgs 
        if self.iter >= self.args.epoch and self.args.l2_reg > 0:
            l2_reg = 0.0 
            for name, p in self.dnn.named_parameters():
                if 'coeff' in name and not self.args.enable_coeff_l2_reg:
                    continue 
                if p.requires_grad:
                    l2_reg += 0.5 * p.square().sum()
            loss += self.args.l2_reg * l2_reg
    
        if (self.args.llaf or self.args.linearpool) and self.args.use_recovery:
            coeff_term = [torch.exp(torch.mean(self.dnn.layers[i].coeff2)) for i in range(len(self.dnn.layers)) if isinstance(self.dnn.layers[i], (LLAF, MyLinearPool))]
   
            recovery_term = len(coeff_term) / sum(coeff_term)
   
            loss += self.args.recovery_weight * recovery_term
        else:
            recovery_term = 0

        if loss.requires_grad:
            loss.backward()
            if self.args.clip > 0.0:
                # torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), self.args.clip)
                if self.args.coeff_clip_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), self.args.clip)
                elif self.args.coeff_clip_type == 'value':
                    torch.nn.utils.clip_grad_value_(self.dnn.parameters(), self.args.clip)
                else:
                    raise NotImplementedError
            if self.args.coeff_clip > 0.0:
                if verbose and self.iter % self.args.print_freq == 0:
                    pre_grad_norm = 0
                    
                    for p in self.optimizer.param_groups[1]['params']:
                        # print(p.shape)
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            pre_grad_norm += param_norm.item() ** 2
                    pre_grad_norm = pre_grad_norm ** 0.5
                if self.args.coeff_clip_type == 'norm':
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[1]['params'], self.args.coeff_clip)
                elif self.args.coeff_clip_type == 'value':
                    torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[1]['params'], self.args.coeff_clip)
                else:
                    raise NotImplementedError
                if verbose and self.iter % self.args.print_freq == 0:
                    post_grad_norm = 0
                    for p in self.optimizer.param_groups[1]['params']:
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            post_grad_norm += param_norm.item() ** 2
                    post_grad_norm = post_grad_norm ** 0.5
                    self.logger.info(f'pre {pre_grad_norm}, post {post_grad_norm}')

        grad_norm = 0
        for name, p in self.dnn.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if step:
            self.optimizer.step()
            if self.args.sep_optim:
                self.optimizer_coeff.step()
        if verbose:
            if (self.iter < self.args.epoch and self.iter % 100== 0) or (self.iter >= self.args.epoch and self.iter % self.args.print_freq == 0):
                for loss_u_i, loss_b_i, loss_f_i, nu, rho, beta in zip(loss_u_list, loss_b_list, loss_f_list, self.nu, self.rho, self.beta):
                    loss_i = self.L_u*loss_u_i + self.L_b*loss_b_i + self.L_f*loss_f_i
                    self.logger.info(
                        'epoch %d, nu: %.5e, rho: %.5e, beta: %.5e, gradient: %.5e, loss: %.5e, recovery: %.5e, loss_u: %.5e, L_u: %.5e, loss_b: %.5e, L_b: %.5e, loss_f: %.5e, L_f: %.5e' % (self.iter, nu, rho, beta, grad_norm, loss_i.item(), recovery_term, loss_u_i.item(), self.L_u, loss_b_i.item(), self.L_b, loss_f_i.item(), self.L_f)
                    )
                  
                    
                    self.writer.add_scalars(f'nu{nu}_rho{rho}_beta{beta}_loss_all', {'loss_all':loss_i.item()}, self.iter)
                    self.writer.add_scalars(f'nu{nu}_rho{rho}_beta{beta}_loss_u', {'loss_u':loss_u_i.item()}, self.iter)
                    self.writer.add_scalars(f'nu{nu}_rho{rho}_beta{beta}_loss_b', {'loss_b':loss_b_i.item()}, self.iter)
                    self.writer.add_scalars(f'nu{nu}_rho{rho}_beta{beta}_loss_f', {'loss_f':loss_f_i.item()}, self.iter)
                    if self.args.activation == 'rational':
                        for name, para in self.dnn.named_parameters():
                            if 'denominator' in name or 'numerator' in name:
                                self.logger.info(f'{name} {para}')
                    if self.args.linearpool or self.args.llaf:
                        for name, para in self.dnn.named_parameters():
                            if 'coeff' in name:
                                if 'coeff2' in name:
                                    para_log = para.mean(dim=-1)
                                    # self.logger.info(f'{name} {para.mean(dim=-1)}')
                                elif self.args.aggregate == 'sigmoid':
                                    para_log = torch.sigmoid(para).mean(-1) 
                                    # self.logger.info(f'{name} {torch.sigmoid(para).mean(-1)}')
                                elif self.args.aggregate == 'softmax':
                                    para_log = torch.softmax(para/self.args.tau,dim=0).mean(-1)
                                    # self.logger.info(f'{name} {torch.softmax(para/self.args.tau,dim=0).mean(-1)}')
                                elif self.args.aggregate == 'unlimited':
                                    para_log = para.mean(-1)
                                    # self.logger.info(f'{name} {para.mean(-1)}')
                                else:
                                    para_log = (para / para.abs().sum(dim=0)).mean(-1)
                                    # self.logger.info(f'{name} {(para / para.abs().sum(dim=0)).mean(-1)}')
                                self.logger.info(f'{name} {para_log}')
                                
                                if 'coeff2' not in name and not self.args.channel_wise:
                                    self.logger.info(f'{name} {para.flatten().data}')
                                    for ele, para_log_ele in enumerate(para_log):
                                        self.writer.add_scalars(f'{name}', {f'{ele}': para_log_ele.item()}, self.iter)
                                        self.writer.add_scalars(f'{name}_value', {f'{ele}': para[ele].item()}, self.iter)
                
          
            if (self.iter > 0 and  self.iter < self.args.epoch and self.iter % 1000 == 0) or (self.iter >= self.args.epoch and self.iter % self.args.valid_freq == 0):
                self.validate()
                if self.iter >= self.args.epoch:
                    if self.args.plot_loss:
                        self.draw_loss(name=f'{self.repeat}_{self.iter}_')
            
            self.iter += 1
        return loss

    def train_adam(self, adam_lr, epoch, X_star, u_star):
        # import pdb 
        # pdb.set_trace()
        if self.args.linearpool or self.args.llaf:
            params_net = []
            params_net_first_layer = []
            params_net_second_layer = []
            params_coeff = []
            params_coeff_first_layer = []
            for name, param in self.dnn.named_parameters():
                if 'coeff' in name:
                    if 'layer_0' in name:
                        params_coeff_first_layer.append(param)
                    else:
                        params_coeff.append(param)
                else:
                    # if 'layer_0' in name or 'layer_1' in name:
                    if 'layer_0' in name:
                        params_net_first_layer.append(param)
                        # params_coeff_first_layer.append(param)
                    elif 'layer_1' in name:
                        params_net_second_layer.append(param)
                    else:
                        params_net.append(param)
            if not self.args.sep_optim:
                self.optimizer = choose_optimizer('AdamW', [{'params': params_net}, {'params': params_net_first_layer, 'lr': self.args.lr_first_layer}, {'params': params_net_second_layer, 'lr': self.args.lr_second_layer}, {'params': params_coeff, 'lr': self.args.coeff_lr, 'betas':(self.args.coeff_beta1, self.args.coeff_beta2)}, {'params': params_coeff_first_layer, 'lr': self.args.coeff_lr_first_layer, 'betas':(self.args.coeff_beta1, self.args.coeff_beta2)}], adam_lr, weight_decay=self.args.weight_decay)
            else:
                self.optimizer = choose_optimizer('AdamW', [{'params': params_net}, {'params': params_net_first_layer, 'lr': self.args.lr_first_layer}, {'params': params_net_second_layer, 'lr': self.args.lr_second_layer}], adam_lr, weight_decay=self.args.weight_decay)
                # self.optimizer = choose_optimizer('Adam', [{'params': params_net}], adam_lr)
                if self.args.sep_optimizer == 'sgd':
                    self.optimizer_coeff = torch.optim.SGD([{'params': params_coeff}, {'params': params_coeff_first_layer, 'lr': self.args.coeff_lr_first_layer}], self.args.coeff_lr, momentum=self.args.momentum)
                elif self.args.sep_optimizer == 'adam':
                    self.optimizer_coeff = torch.optim.Adam([{'params': params_coeff}, {'params': params_coeff_first_layer, 'lr': self.args.coeff_lr_first_layer}], self.args.coeff_lr, betas=(self.args.coeff_beta1, self.args.coeff_beta2), weight_decay=self.args.coeff_weight_decay)
                else:
                    raise NotImplementedError 
        
        else:
            if self.args.lr_first_layer>0:
                params_net = []
                params_net_first_layer = []

                for name, param in self.dnn.named_parameters():
                    if 'layer_0' in name:
                        params_net_first_layer.append(param)
                        # params_coeff_first_layer.append(param)
                    else:
                        params_net.append(param)
                self.optimizer = choose_optimizer('AdamW', [{'params': params_net}, {'params': params_net_first_layer, 'lr': self.args.lr_first_layer}], adam_lr, weight_decay=self.args.weight_decay)
            else:
                self.logger.info('use only one optimizer')
                self.optimizer = choose_optimizer('AdamW', self.dnn.parameters(), adam_lr, weight_decay=self.args.weight_decay)
           
        warm_up_iter = self.args.warm_up_iter
        lr_min = 1e-3
        lr_max = 1 
        T_max = epoch if self.args.T_max == 0 else self.args.T_max 
        if self.args.cosine_decay:
            if self.args.constant_warmup > 0:
                lambda0 = lambda cur_iter: self.args.constant_warmup if cur_iter < warm_up_iter else \
                    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
            else:
                lambda0 = lambda cur_iter: 1e-3 + (lr_max/2**(cur_iter//T_max) - 1e-3) * (cur_iter%T_max) / warm_up_iter if  (cur_iter%T_max) < warm_up_iter else \
                    (lr_min + 0.5*(lr_max/2**(cur_iter//T_max)-lr_min)*(1.0+math.cos(((cur_iter%T_max)-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
        else:
            lambda0 = lambda cur_iter: 1e-3 + (1 - 1e-3) * cur_iter / warm_up_iter if  cur_iter < warm_up_iter else 1.0
        schduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda0)
        if self.args.sep_optim:
            sep_warm_up_iter = self.args.sep_warm_up_iter
            lr_min = 1e-3
            lr_max = 1.0 
            T_max = epoch 
            if self.args.sep_cosine_decay:
                if self.args.constant_warmup > 0:
                    sep_lambda0 = lambda cur_iter: self.args.constant_warmup if cur_iter < sep_warm_up_iter else \
                        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-sep_warm_up_iter)/(T_max-sep_warm_up_iter)*math.pi)))
                else:
                    sep_lambda0 = lambda cur_iter: 1e-3 + (1 - 1e-3) * cur_iter / sep_warm_up_iter if  cur_iter < sep_warm_up_iter else \
                        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-sep_warm_up_iter)/(T_max-sep_warm_up_iter)*math.pi)))
            else:
                sep_lambda0 = lambda cur_iter: 1e-3 + (1 - 1e-3) * cur_iter / sep_warm_up_iter if  cur_iter < sep_warm_up_iter else 1.0
            scheduler_coeff = torch.optim.lr_scheduler.LambdaLR(self.optimizer_coeff, lr_lambda=sep_lambda0)
        # schduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epoch, eta_min=1e-3*adam_lr)
        self.dnn.train() 
        self.logger.info('>>>>>> Adam optimizer')

        self.args.pre_use_recovery = self.args.use_recovery 
        
        for epoch_n in range(epoch):
            loss = self.loss_pinn(verbose=True,step=True)
            # self.logger.info(f"lr {self.optimizer.param_groups[0]['lr']}")
            # self.writer.add_scalars('lr', {'pg0':self.optimizer.param_groups[0]['lr']}, self.iter)
            
            if (epoch_n+1) % 10000 == 0:
                if self.args.plot_loss:
                    self.draw_loss(name=f'{self.repeat}_{self.iter}_')
            if not self.args.not_adapt_adam_lr:
                schduler.step()
                if self.args.sep_optim:
                    scheduler_coeff.step()
      
        torch.save({'state_dict':self.dnn.state_dict()}, os.path.join(self.args.work_dir,f'model_adam_{self.repeat}.pth.tar') )
        if self.args.checkpoint != '':
            self.logger.info(f'load model from {self.args.checkpoint}')
            checkpoint = torch.load(self.args.checkpoint)
            self.dnn.load_state_dict(checkpoint['state_dict'])
        
   
        
        self.logger.info(f'>>>>>> {self.optimizer_name} optimizer')
       
       
        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr, max_iter=self.args.max_iter, line_search_fn=self.args.line_search_fn)
        self.dnn.train()
        if not self.args.disable_lbfgs:
            self.optimizer.step(self.loss_pinn)
        torch.save({'state_dict':self.dnn.state_dict()}, os.path.join(self.args.work_dir,f'model_lbfgs_{self.repeat}.pth.tar') )
        
       
        if self.args.plot_loss:
            self.draw_loss(name=str(self.repeat))
       
        return self.loss_pinn(evaluate=True)

    def test_loss_f(self, X_f_train):
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).double().to(self.device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).double().to(self.device)
        self.dnn.train()
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_f = f_pred ** 2
        self.logger.info(f'{loss_f.mean():.5e}')
        return loss_f.detach().cpu().numpy()

    def draw_loss(self, name=''):
        self.dnn.train()
        u_pred = self.net_u(self.x_u, self.t_u)
        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
        if 'CH' in self.system and self.args.decouple:
            u_pred = [u_i[:, 0:1] for u_i in u_pred]
            u_pred_lb = [u_i[:, 0:1] for u_i in u_pred_lb]
            u_pred_ub = [u_i[:, 0:1] for u_i in u_pred_ub]
        x_f = torch.tensor(self.X_star[:, 0:1], requires_grad=True).double().to(self.device)
        t_f = torch.tensor(self.X_star[:, 1:2], requires_grad=True).double().to(self.device)
        
        # iter_num = math.ceil(x_f.shape[0] / self.args.N_f)
        iter_num = 0
        f_pred = [list() for _ in range(self.num_head)]
        u_t_plot = [list() for _ in range(self.num_head)]
        u_x_plot = [list() for _ in range(self.num_head)]
        while(iter_num < x_f.shape[0]):
            f_pred_i, u_tx_i = self.net_f(x_f[int(iter_num):int(iter_num+self.args.N_f)], t_f[int(iter_num):int(iter_num+self.args.N_f)], return_gradient=True)
            iter_num += self.args.N_f 
            for f_list, f_pred_i_head in zip(f_pred, f_pred_i):
                f_list.append(f_pred_i_head.detach())
                del f_pred_i_head
            for u_t_list, u_x_list, (u_t_i, u_x_i) in zip(u_t_plot, u_x_plot, u_tx_i):
                u_t_list.append(u_t_i)
                u_x_list.append(u_x_i)
                del u_t_i, u_x_i 
            # f_pred.append(f_pred_i.detach())
        
        f_pred = [torch.cat(f_list) for f_list in f_pred]
        if 'CH' in self.system and self.args.decouple:
            # f_pred = [f_pred_i.abs().sum(dim=-1) for f_pred_i in f_pred]
            f_pred1 = [f_pred_i[:,0:1] for f_pred_i in f_pred]
            f_pred2 = [f_pred_i[:,1:2] for f_pred_i in f_pred]
        u_t_plot = [torch.cat(u_t_list) for u_t_list in u_t_plot]
        u_x_plot = [torch.cat(u_x_list) for u_x_list in u_x_plot]
        u_pred_f = self.net_u(x_f, t_f)
        if 'CH' in self.system and self.args.decouple:
            u_pred_f = [u_i[:, 0:1] for u_i in u_pred_f]
        
        for idx in range(self.num_head):
            if 'CH' in self.system and self.args.decouple:
                loss_u = (self.u[:,0:1] - u_pred[idx]).abs()
            else:
                loss_u = (self.u - u_pred[idx]).abs()
            loss_u = loss_u.detach().cpu().numpy() 
            # import pdb 
            # pdb.set_trace()
            loss_b = (u_pred_lb[idx] - u_pred_ub[idx]).abs()
            if self.nu[idx] != 0:
                u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb[idx], u_pred_ub[idx], self.x_bc_lb, self.x_bc_ub)
                loss_b += (u_pred_lb_x - u_pred_ub_x).abs()
            u_pred_lb_i = u_pred_lb[idx].detach().cpu().numpy() 
            u_pred_ub_i = u_pred_ub[idx].detach().cpu().numpy() 
            loss_b = loss_b.detach().cpu().numpy() 
            u_b = self.Exact[idx][:, 0:1]
            diff_lb = np.abs(u_b - u_pred_lb_i)
            diff_ub = np.abs(u_b - u_pred_ub_i)
            fig, axes = plt.subplots(3,1,figsize=(9, 12))

            axes[0].plot(self.x_u.reshape(-1).detach().cpu().numpy(), loss_u.reshape(-1))
            axes[0].set_title('loss_initial')
            axes[1].plot(self.t_bc_lb.reshape(-1).detach().cpu().numpy(), loss_b.reshape(-1))
            axes[1].set_title('loss_boundary')
            axes[2].plot(self.t_bc_lb.reshape(-1).detach().cpu().numpy(), diff_lb.reshape(-1), c='r')
            axes[2].plot(self.t_bc_lb.reshape(-1).detach().cpu().numpy(), diff_ub.reshape(-1), c='b')
            axes[2].legend(['lower bound', 'upper bound'])
            axes[2].set_title('error_boundary')
            plt.savefig(os.path.join(self.args.work_dir, name+f'loss_initial_boundary_nu{self.nu[idx]}_rho{self.rho[idx]}_beta{self.beta[idx]}.png'))
            plt.close(fig)

            
            
            if 'CH' in self.system and self.args.decouple:
                loss_f = (f_pred1[idx].abs()).detach().cpu().numpy().reshape(self.args.nt, self.args.xgrid)
                loss_f2 = (f_pred2[idx].abs()).detach().cpu().numpy().reshape(self.args.nt, self.args.xgrid)
                loss_f[0,:] = 0
                loss_f2[0,:] = 0
                fig = plt.figure(figsize=(9, 6))
                ax0 = fig.add_subplot(111)
                h0 = ax0.imshow(loss_f2.T, interpolation='nearest', cmap='rainbow',
                            extent=[t_f.detach().cpu().numpy().min(), t_f.detach().cpu().numpy().max(), x_f.detach().cpu().numpy().min(), x_f.detach().cpu().numpy().max()],
                            origin='lower', aspect='auto')
                divider0 = make_axes_locatable(ax0)
                cax0 = divider0.append_axes("right", size="5%", pad=0.10)
                cbar0 = fig.colorbar(h0, cax=cax0)
                cbar0.ax.tick_params(labelsize=15)

                ax0.set_xlabel('t', fontweight='bold', size=15)
                ax0.set_ylabel('x', fontweight='bold', size=15)
                ax0.set_title('loss_pde2')
                plt.savefig(os.path.join(self.args.work_dir, name+f'loss_pde2_nu{self.nu[idx]}_rho{self.rho[idx]}_beta{self.beta[idx]}.png'))
                plt.close(fig)
            else:
                loss_f = (f_pred[idx].abs()).detach().cpu().numpy().reshape(self.args.nt, self.args.xgrid)
                loss_f[0,:] = 0
            u_pred_f_i = u_pred_f[idx].reshape(self.args.nt, self.args.xgrid).detach().cpu().numpy()
            diff_u = np.abs(self.Exact[idx]-u_pred_f_i)
            fig = plt.figure(figsize=(9, 18))
            ax0 = fig.add_subplot(311)

            h0 = ax0.imshow(u_pred_f_i.T, interpolation='nearest', cmap='rainbow',
                            extent=[t_f.detach().cpu().numpy().min(), t_f.detach().cpu().numpy().max(), x_f.detach().cpu().numpy().min(), x_f.detach().cpu().numpy().max()],
                            origin='lower', aspect='auto')
            divider0 = make_axes_locatable(ax0)
            cax0 = divider0.append_axes("right", size="5%", pad=0.10)
            cbar0 = fig.colorbar(h0, cax=cax0)
            cbar0.ax.tick_params(labelsize=15)
            ax0.set_title('predition')

            ax0.set_xlabel('t', fontweight='bold', size=15)
            ax0.set_ylabel('x', fontweight='bold', size=15)
            
            ax1 = fig.add_subplot(312)

            h1 = ax1.imshow(diff_u.T, interpolation='nearest', cmap='rainbow',
                            extent=[t_f.detach().cpu().numpy().min(), t_f.detach().cpu().numpy().max(), x_f.detach().cpu().numpy().min(), x_f.detach().cpu().numpy().max()],
                            origin='lower', aspect='auto')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.10)
            cbar1 = fig.colorbar(h1, cax=cax1)
            cbar1.ax.tick_params(labelsize=15)
            ax1.set_title('error')

            ax1.set_xlabel('t', fontweight='bold', size=15)
            ax1.set_ylabel('x', fontweight='bold', size=15)

            ax2 = fig.add_subplot(313)

            h2 = ax2.imshow(loss_f.T, interpolation='nearest', cmap='rainbow',
                            extent=[t_f.detach().cpu().numpy().min(), t_f.detach().cpu().numpy().max(), x_f.detach().cpu().numpy().min(), x_f.detach().cpu().numpy().max()],
                            origin='lower', aspect='auto')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.10)
            cbar2 = fig.colorbar(h2, cax=cax2)
            cbar2.ax.tick_params(labelsize=15)

            ax2.set_xlabel('t', fontweight='bold', size=15)
            ax2.set_ylabel('x', fontweight='bold', size=15)
            ax2.set_title('loss_pde')

            plt.savefig(os.path.join(self.args.work_dir, name+f'loss_pde_nu{self.nu[idx]}_rho{self.rho[idx]}_beta{self.beta[idx]}.png'))
            plt.close(fig)

            fig = plt.figure(figsize=(9, 18))
            ax0 = fig.add_subplot(211)

            u_t_pred = u_t_plot[idx].reshape(self.args.nt, self.args.xgrid).detach().cpu().numpy()
            u_x_pred = u_x_plot[idx].reshape(self.args.nt, self.args.xgrid).detach().cpu().numpy()
            h0 = ax0.imshow(u_t_pred.T, interpolation='nearest', cmap='binary',
                            extent=[t_f.detach().cpu().numpy().min(), t_f.detach().cpu().numpy().max(), x_f.detach().cpu().numpy().min(), x_f.detach().cpu().numpy().max()],
                            origin='lower', aspect='auto')
            divider0 = make_axes_locatable(ax0)
            cax0 = divider0.append_axes("right", size="5%", pad=0.10)
            cbar0 = fig.colorbar(h0, cax=cax0)
            cbar0.ax.tick_params(labelsize=15)
            ax0.set_title('u_t')

            ax0.set_xlabel('t', fontweight='bold', size=15)
            ax0.set_ylabel('x', fontweight='bold', size=15)
            
            ax1 = fig.add_subplot(212)

            h1 = ax1.imshow(u_x_pred.T, interpolation='nearest', cmap='binary',
                            extent=[t_f.detach().cpu().numpy().min(), t_f.detach().cpu().numpy().max(), x_f.detach().cpu().numpy().min(), x_f.detach().cpu().numpy().max()],
                            origin='lower', aspect='auto')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.10)
            cbar1 = fig.colorbar(h1, cax=cax1)
            cbar1.ax.tick_params(labelsize=15)
            ax1.set_title('u_x')

            ax1.set_xlabel('t', fontweight='bold', size=15)
            ax1.set_ylabel('x', fontweight='bold', size=15)

            plt.savefig(os.path.join(self.args.work_dir, name+f'gradient_nu{self.nu[idx]}_rho{self.rho[idx]}_beta{self.beta[idx]}.png'))
            plt.close(fig)

    def validate(self):
        self.dnn_auxiliary.load_state_dict(self.dnn.state_dict())
        u_pred = self.predict(self.X_star, return_all=True, use_auxiliary=True)
        f_pred = self.evaluate_loss_f(self.X_star, return_all=True, use_auxiliary=True)

    
        for u_pred_i, f_pred_i, Exact, nu, beta, rho in zip(u_pred, f_pred, self.Exact, self.nu, self.beta, self.rho):
            u_star = Exact.reshape(-1, 1)

            error_u_relative = np.linalg.norm(u_star-u_pred_i, 2)/np.linalg.norm(u_star, 2)
    
            error_u_abs = np.mean(np.abs(u_star - u_pred_i))
            error_u_linf = np.linalg.norm(u_star - u_pred_i, np.inf)/np.linalg.norm(u_star, np.inf)
            error_f_test = np.mean(f_pred_i ** 2)
            self.logger.info(f"lr {self.optimizer.param_groups[0]['lr']}")
            if self.args.sep_optim:
                self.logger.info(f"lr {self.optimizer_coeff.param_groups[0]['lr']}")
            self.logger.info(f'Head for nu {nu}, rho {rho}, beta {beta}')
            self.logger.info('Error u rel: %e' % (error_u_relative))
            self.logger.info('Error u abs: %e' % (error_u_abs))
            self.logger.info('Error u linf: %e' % (error_u_linf))
            self.logger.info('Loss f test: %e' % (error_f_test))
            # import pdb
            # pdb.set_trace()
            self.writer.add_scalars('error', {'loss_F':error_f_test}, self.iter)
            self.writer.add_scalars('error', {'relative':error_u_relative}, self.iter)
       

    def predict(self, X, return_all=False, use_auxiliary=False):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).double().to(self.device)
        if use_auxiliary:
            self.dnn_auxiliary.eval()
      
        u = self.net_u(x, t, use_auxiliary=use_auxiliary)
        if 'CH' in self.system and self.args.decouple:
            u = [u_i[:, 0:1].detach().cpu().numpy() for u_i in u]
        else:
            u = [u_i.detach().cpu().numpy() for u_i in u]
        if return_all:
            return u
        else:
            return u[-1]
    
    def evaluate_loss_f(self, X, return_all=False, use_auxiliary=False):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).double().to(self.device)
        iter_num = 0
        f_pred = [list() for _ in range(self.num_head)]
        while(iter_num < x.shape[0]):
            f_pred_i  = self.net_f(x[int(iter_num):int(iter_num+self.args.N_f)], t[int(iter_num):int(iter_num+self.args.N_f)], use_auxiliary=use_auxiliary)
            iter_num += self.args.N_f 
            for f_list, f_pred_i_head in zip(f_pred, f_pred_i):
                f_list.append(f_pred_i_head.detach().cpu().numpy())
                del f_pred_i_head
        f_pred = [np.vstack(f_list) for f_list in f_pred]
 
        # f_pred = self.net_f(x, t, use_auxiliary=use_auxiliary)
        # f_pred = [f_pred_i.detach().cpu().numpy() for f_pred_i in f_pred]
        if return_all:
            return f_pred
        else:
            return f_pred[-1]

    