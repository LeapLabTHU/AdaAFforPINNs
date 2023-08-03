"""Run PINNs for convection/reaction/reaction-diffusion with periodic boundary conditions."""

import argparse
import numpy as np
import os
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
import sys 
from scipy.io import loadmat
import h5py 
from net import *
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float64)
# torch.use_deterministic_algorithms(True)
# torch.set_deterministic(True)
################
# Arguments
################
parser = argparse.ArgumentParser(description='Adaptive AF for PINNs')

parser.add_argument('--launcher', default='', type=str)
parser.add_argument('--port', default=29051, type=int)
parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--N_f_x', type=int, default=64, help='Number of collocation points to sample.')
parser.add_argument('--N_f_t', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--ratio', type=float, default=1.0, help='ratio for N_f')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L_f', type=float, default=1.0, help='Multiplier on loss f.')
parser.add_argument('--L_u', type=float, default=1.0, help='Multiplier on loss u.')
parser.add_argument('--L_b', type=float, default=1.0, help='Multiplier on loss b.')
parser.add_argument('--adam_lr', type=float, default=0.0, help='Learning rate for adam.')
parser.add_argument('--coeff_lr', type=float, default=1e-3, help='Learning rate for adam.')
parser.add_argument('--epoch', type=int, default=1000, help='Epoch for adam')
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--start_repeat', type=int, default=0)
parser.add_argument('--sample_type', type=str, default='grid')
parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=str, default='0', help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
parser.add_argument('--rho', type=str, default='0', help='reaction coefficient for u*(1-u) term.')
parser.add_argument('--beta', type=str, default='0', help='beta value that scales the du/dx term. 0 if only doing diffusion.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')

# params for linearpool
parser.add_argument('--linearpool', action='store_true')
parser.add_argument('--llaf', action='store_true')
parser.add_argument('--use_recovery', action='store_true')
parser.add_argument('--scaler', type=float, default=1.0)
parser.add_argument('--poolsize', type=str, default='0')
parser.add_argument('--aggregate', type=str, default='sum')
parser.add_argument('--weight_sharing', action='store_true')
parser.add_argument('--use_auxiliary', action='store_true')
parser.add_argument('--sample_iter', type=int, default=1)
parser.add_argument('--plot_loss', action='store_true')
parser.add_argument('--evaluate', action='store_true')

parser.add_argument('--num_head', type=int, default=1)
parser.add_argument('--not_adapt_adam_lr', action='store_true')
parser.add_argument('--sample_stage', type=int, default=1)
parser.add_argument('--init', action='store_true')
parser.add_argument('--visualize', default=False, help='Visualize the solution.')
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--work_dir', default='debug')
parser.add_argument('--sub_name', default='')
parser.add_argument('--hard_ibc', action='store_true')
parser.add_argument('--channel_wise', action='store_true')
parser.add_argument('--warm_up_iter', type=int, default=1000)
parser.add_argument('--uniform_sample', action='store_true')
parser.add_argument('--use_norm', action='store_true')
parser.add_argument('--exp_alpha', type=float, default=2.0)
parser.add_argument('--sin_alpha', type=float, default=1.0)
parser.add_argument('--fix_sample', action='store_true')
parser.add_argument('--cosine_decay', action='store_true')
parser.add_argument('--disable_lbfgs', action='store_true')

parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--clip', type=float, default=0.0)
parser.add_argument('--coeff_clip', type=float, default=0.0)
parser.add_argument('--coeff_clip_type', type=str, default='norm')
parser.add_argument('--recovery_weight', type=float, default=1.0)
parser.add_argument('--enable_scaling', action='store_true')
parser.add_argument('--detach_u', action='store_true')
parser.add_argument('--detach_b', action='store_true')
parser.add_argument('--detach_f', action='store_true')
parser.add_argument('--coeff_beta1', type=float, default=0.9)
parser.add_argument('--coeff_beta2', type=float, default=0.999)
parser.add_argument('--constant_warmup', type=float, default=0.0)
parser.add_argument('--sep_optim', action='store_true')
parser.add_argument('--sep_warm_up_iter', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--sep_optimizer', type=str, default='sgd')
parser.add_argument('--coeff_lr_first_layer', type=float, default=1e-3)
parser.add_argument('--lr_first_layer', type=float, default=1e-3)
parser.add_argument('--lr_second_layer', type=float, default=1e-3)
parser.add_argument('--sep_cosine_decay', action='store_true')
parser.add_argument('--target_seed', type=int, default=None)
parser.add_argument('--print_freq', type=int, default=1000)
parser.add_argument('--valid_freq', type=int, default=1000)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--coeff_weight_decay', type=float, default=0.0)
parser.add_argument('--extra_N_f', type=int, default=0)
parser.add_argument('--range', type=float, default=0.01)
parser.add_argument('--enable_coeff_l2_reg', action='store_true')


parser.add_argument('--T_max', type=int, default=0)
parser.add_argument('--line_search_fn', type=str, default=None)
parser.add_argument('--gain_0', type=float, default=1.0)
parser.add_argument('--gain', type=float, default=1.0)
parser.add_argument('--enable_vx', action='store_true')
parser.add_argument('--enable_vt', action='store_true')
parser.add_argument('--include_t0', action='store_true')

parser.add_argument('--max_iter', type=int, default=15000)

# for CH 
parser.add_argument('--decouple', action='store_true')
parser.add_argument('--high_bc', action='store_true')
parser.add_argument('--four_order', action='store_true')
parser.add_argument('--enable_adaptive_tol', action='store_true')
parser.add_argument('--fine_grid', action='store_true')
parser.add_argument('--segment', type=int, default=-1)


parser.add_argument('--dtype_float32', action='store_true')
parser.add_argument('--exp_dir', default='./')
parser.add_argument('--random_init', action='store_true', help='random init for coeff')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--taylor', action='store_true')
parser.add_argument('--taylor_scale', action='store_true')
parser.add_argument('--taylor_order', type=int, default=0)


args = parser.parse_args()

if not args.dtype_float32:
    torch.set_default_dtype(torch.float64)

args.name = args.work_dir
logger = init_environ(args)
logger.info(args.name)
logger.info(args)

# CUDA support
if args.gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

nu = [float(item) for item in args.nu.split(',')]
beta = [float(item) for item in args.beta.split(',')]
rho = [float(item) for item in args.rho.split(',')]


# args.name = os.path.join(args.work_dir, args.name)


# parse the layers list here
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

############################
# Process data
############################

if args.data_path is not None:
    # for burger, AC, KdV and CH 
    if args.system == 'burger':
        data = np.load(args.data_path)
        t, x, Exact = data["t"], data["x"], data["usol"].T
     
    elif args.system == 'AC':
        # from scipy.io import loadmat
        data = loadmat(args.data_path)
        t,x,Exact = data["t"], data["x"], data["u"] # (1,101), (1,201), (101,201)
        t = t.reshape(-1)
        x = x.reshape(-1)
    elif args.system == 'KdV':
        # from scipy.io import loadmat
        data = loadmat(args.data_path)
        t,x,Exact = data["tt"], data["x"], data["uu"].T # (1,201), (1,512), (512,201)
        t = t.reshape(-1)
        x = x.reshape(-1)
      
        x = np.append(x,1.0)
        Exact_init = Exact[:, 0:1]
        Exact = np.hstack([Exact, Exact_init])
  
    elif args.system == 'CH':
        data = loadmat(args.data_path)
        t,x,Exact = data["tt"], data["x"], data["uu"].T # (1,201), (1,512), (512,201)
        u22 = data['u22'] # (512,1)
        t = t.reshape(-1)
        # x = x.reshape(-1)
        x = np.linspace(-1,1,512,endpoint=False)
        x = np.append(x,1.0)
        Exact_init = Exact[:, 0:1]
        Exact = np.hstack([Exact, Exact_init])
        if args.segment > 0:
            t = t[:args.segment]
            Exact = Exact[:args.segment, :]

      
    X, T = np.meshgrid(x, t)
    X_star = np.vstack((np.ravel(X), np.ravel(T))).T
    u_star = Exact.flatten()[:, None] 
    Exact_list = [Exact]

   
    if args.sample_type == 'grid':
        if args.include_t0:
            x_noboundary = np.linspace(-1, 1, args.N_f_x, endpoint=False).reshape(-1, 1) # not inclusive
            t_noinitial = np.linspace(0, 1, args.N_f_t).reshape(-1, 1)
        else:
            x_noboundary = np.linspace(-1, 1, args.N_f_x+1, endpoint=False).reshape(-1, 1)[1:] # not inclusive
            if args.fine_grid:
                t_noinitial = np.linspace(0,0.05,50+1,endpoint=False)[1:]
                t_noinitial2 = np.linspace(0.05, 1, args.N_f_t)
                t_noinitial = np.hstack([t_noinitial, t_noinitial2])
                args.N_f_t = args.N_f_t + 50 
            else:
                t_noinitial = np.linspace(0, 1, args.N_f_t+1).reshape(-1, 1)[1:]
        
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))
        logger.info(f'sample from grid: {args.N_f_x} {args.N_f_t}')
        
    else:
        X_star_noinitial_noboundary = {'lb':[x.min(), t.min()], 'ub':[x.max(), t.max()]}
        logger.info(f'sample given interval: {X_star_noinitial_noboundary}')
        # X_f_train= sample_random_interval(X_star_noinitial_noboundary['lb'],X_star_noinitial_noboundary['ub'], args.N_f)

    # sample collocation points only from the interior (where the PDE is enforced)
    set_seed(args.seed)
    # X_f_train, idx = sample_random_type(X_star_noinitial_noboundary, args.N_f)
    X_f_train, idx = sample_random_type(X_star_noinitial_noboundary, args.N_f, args.extra_N_f, args.range)
    args.N_f = args.N_f + args.extra_N_f 
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1,:].T # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = -1, and 

    # generate the other BC, now at x=1
    bc_ub = np.hstack((X[:,-1:], T[:,-1:]))
    u_train = uu1 # just the initial condition
    X_u_train = xx1 # (x,t) for initial condition

    G = np.full(1, float(args.source))
else:
    # for convection, reaction, reaction-diffusion 
    x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

    if args.sample_type == 'grid':
        # # remove initial and boundaty data from X_star
        # t_noinitial = t[1:]
        # # remove boundary at x=0
        # x_noboundary = x[1:-1]
        if args.include_t0:
            x_noboundary = np.linspace(0, 2*np.pi, args.N_f_x, endpoint=False).reshape(-1, 1) # not inclusive
            t_noinitial = np.linspace(0, 1, args.N_f_t).reshape(-1, 1)
        else:
            x_noboundary = np.linspace(0, 2*np.pi, args.N_f_x+1, endpoint=False).reshape(-1, 1)[1:] # not inclusive
            t_noinitial = np.linspace(0, 1, args.N_f_t+1).reshape(-1, 1)[1:]
        
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))
        logger.info(f'sample from grid: {args.N_f_x} {args.N_f_t}')
    else:
        X_star_noinitial_noboundary = {'lb':[x.min(), t.min()], 'ub':[x.max(), t.max()]}
        logger.info(f'sample given interval: {X_star_noinitial_noboundary}')

    # sample collocation points only from the interior (where the PDE is enforced)
    set_seed(args.seed)
    X_f_train, idx = sample_random_type(X_star_noinitial_noboundary, args.N_f)

    u_vals_list = list()
    for nu_i, beta_i, rho_i in zip(nu, beta, rho):
        if 'convection' in args.system or 'diffusion' in args.system:
            u_vals = convection_diffusion(args.u0_str, nu_i, beta_i, args.source, args.xgrid, args.nt)
           
        elif 'rd' in args.system:
            u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu_i, rho_i, args.xgrid, args.nt)
           
        elif 'reaction' in args.system:
            u_vals = reaction_solution(args.u0_str, rho_i, args.xgrid, args.nt)
         
        else:
            print("WARNING: System is not specified.")
        u_vals_list.append(u_vals)
    G = np.full(1, float(args.source))

    u_star_list = [u_vals.reshape(-1, 1) for u_vals in u_vals_list] # Exact solution reshaped into (n, 1)
    Exact_list = [u_star.reshape(len(t), len(x)) for u_star in u_star_list] # Exact on the (x,t) grid
    u_star = u_star_list[-1]
    Exact = Exact_list[-1]
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1,:].T # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
    uu2 = Exact[:,0:1] # u(-end, t)

    # generate the other BC, now at x=2pi
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train = uu1 # just the initial condition
    X_u_train = xx1 # (x,t) for initial condition

if args.system != 'CH':
    u22 = None 

layers.insert(0, X_u_train.shape[-1])

############################
# Train the model
############################
repeat = args.repeat 
error_u_relative_all = []
error_u_abs_all = []
error_u_linf_all = []
losses_all = []
losses_u = []
losses_f = []
losses_b = []
losses_f_test = []

linear_pool_coeff = []

# numpy array float64 to float32 
# X_u_train = X_u_train.astype(np.float32)
# u_train = u_train.astype(np.float32)
# X_f_train = X_f_train.astype(np.float32)
# bc_lb = bc_lb.astype(np.float32)
# bc_ub = bc_ub.astype(np.float32)
# X_star = X_star.astype(np.float32)

for i in range(args.start_repeat, repeat):
    
    if args.target_seed is not None and i != args.target_seed:
        continue 
    set_seed(args.seed+i) # for weight initialization
    if not args.fix_sample:
        X_f_train, idx = sample_random_type(X_star_noinitial_noboundary, args.N_f)
        X_f_train = X_f_train[:int(args.ratio*args.N_f)]
    # X_f_train = X_star_noinitial_noboundary
    # X_f_train = X_star
    logger.info(X_f_train)
    logger.info(X_f_train.shape)
    logger.info(f'seed: {args.seed+i}')
    model = PhysicsInformedNN_pbc(args, i, logger, device, args.system, X_star_noinitial_noboundary, X_u_train, u_train, X_f_train, u22, bc_lb, bc_ub, X_star, Exact_list, layers, G, nu, beta, rho,
                                args.optimizer_name, args.lr, args.net, args.activation)
   
    if args.evaluate:
 
        u_pred = model.predict(X_star)
        loss_f = model.evaluate_loss_f(X_star)
        error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        error_u_abs = np.mean(np.abs(u_star - u_pred))
        error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)

        logger.info('Error u rel: %e' % (error_u_relative))
        logger.info('Error u abs: %e' % (error_u_abs))
        logger.info('Error u linf: %e' % (error_u_linf))
        data_dict = {'exact':u_star.reshape(len(t), len(x)), 'pred':u_pred.reshape(len(t), len(x))}
        np.save(os.path.join(args.work_dir, 'results.npy'), data_dict)
        break

  
    loss_all, loss_u, loss_b, loss_f = model.train_adam(args.adam_lr, args.epoch, X_star, u_star)
    losses_all.append(loss_all)
    losses_u.append(loss_u)
    losses_b.append(loss_b)
    losses_f.append(loss_f)

    u_pred = model.predict(X_star)
    f_pred = model.evaluate_loss_f(X_star)

    if args.linearpool:
        coeff_all = []
        for name, para in model.dnn.named_parameters():
            if 'coeff' in name:
                if args.aggregate == 'sigmoid':
                    coeff = torch.sigmoid(para).mean(-1)
                elif args.aggregate == 'softmax':
                    coeff = torch.softmax(para/args.tau,dim=0).mean(-1)
                elif args.aggregate == 'unlimited':
                    coeff = para.mean(-1)
                else:
                    coeff = (para / para.abs().sum(dim=0)).mean(-1)
                coeff_all.append(coeff)
        coeff_all = torch.cat(coeff_all)
        linear_pool_coeff.append(coeff_all)      
    error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
    error_u_abs = np.mean(np.abs(u_star - u_pred))
    error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)
    error_f_test = np.mean(f_pred ** 2)

    logger.info('Error u rel: %e' % (error_u_relative))
    logger.info('Error u abs: %e' % (error_u_abs))
    logger.info('Error u linf: %e' % (error_u_linf))
    logger.info('Loss all: %e' % (loss_all))
    logger.info('Loss u: %e' % (loss_u))
    logger.info('Loss b: %e' % (loss_b))
    logger.info('Loss f: %e' % (loss_f))
    logger.info('Loss f test: %e' % (error_f_test))
    

    error_u_relative_all.append(error_u_relative)
    error_u_abs_all.append(error_u_abs)
    error_u_linf_all.append(error_u_linf)
    losses_f_test.append(error_f_test)

    if i != repeat-1:
        del model 

if args.linearpool:
    linear_pool_coeff = torch.vstack(linear_pool_coeff)
    mean_coeff = linear_pool_coeff.mean(dim=0)
    std_coeff = linear_pool_coeff.std(dim=0)
    coeff_str = [f'{mean_i:.2f}/{std_i:.2f}' for mean_i, std_i in zip(mean_coeff, std_coeff)]
    coeff_str = ', '.join(coeff_str)
    logger.info(coeff_str)
logger.info('Error u rel: mean %e, std %e' % (np.mean(error_u_relative_all), np.std(error_u_relative_all)))
logger.info('Error u abs: mean %e, std %e' % (np.mean(error_u_abs_all), np.std(error_u_abs_all)))
logger.info('Error u linf: mean %e, std %e' % (np.mean(error_u_linf_all), np.std(error_u_linf_all)))
logger.info('loss_all: mean %e, std %e' % (np.mean(losses_all), np.std(losses_all)))
logger.info('loss_u: mean %e, std %e' % (np.mean(losses_u), np.std(losses_u)))
logger.info('loss_b: mean %e, std %e' % (np.mean(losses_b), np.std(losses_b)))
logger.info('loss_f: mean %e, std %e' % (np.mean(losses_f), np.std(losses_f)))
logger.info('loss_f_test: mean %e, std %e' % (np.mean(losses_f_test), np.std(losses_f_test)))

if args.visualize:
    path = os.path.join(args.work_dir, f"heatmap_results/{args.system}")
    if not os.path.exists(path):
        os.makedirs(path)
    u_pred = u_pred.reshape(len(t), len(x))
    if args.evaluate:
        loss_f = loss_f.reshape(len(t), len(x))
        exact_u(loss_f, x, t, nu, beta, rho, orig_layers, args.N_f, args.L_f, args.source, args.u0_str, args.system, path=path)
    else:
        exact_u(Exact, x, t, nu, beta, rho, orig_layers, args.N_f, args.L_f, args.source, args.u0_str, args.system, path=path)

    u_diff(X_f_train,Exact, u_pred, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L_f, args.source, args.lr, args.u0_str, args.system, path=path)
    u_predict(Exact, u_pred, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L_f, args.source, args.lr, args.u0_str, args.system, path=path)

