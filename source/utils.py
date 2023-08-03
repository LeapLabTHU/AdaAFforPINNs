import numpy as np
import random
import torch
import torch.distributed as dist

import os
import time
from pathlib import Path
import logging
import subprocess

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)

def sample_random_type(X_all, N, extra_N=0, range=0):
# def sample_random_type(X_all, N):
    """Given an array of (x,t) points, sample N points from this."""
    # set_seed(seed) # this can be fixed for all N_f
    if isinstance(X_all, dict):
        point_dim = []
        lb = X_all['lb']
        ub = X_all['ub']
        for lb_i, ub_i in zip(lb, ub):
            point_dim.append(np.random.uniform(low=lb_i, high=ub_i, size=(N,1)))
        X_sampled = np.hstack(point_dim)
        if extra_N > 0:
            extra_point = []
            extra_point.append(np.random.uniform(low=-1.0*range, high=range, size=(extra_N,1)))
            extra_point.append(np.random.uniform(low=0, high=1, size=(extra_N,1)))
            extra_X_sampled = np.hstack(extra_point)
            # import pdb 
            # pdb.set_trace()
            X_sampled = np.vstack([X_sampled, extra_X_sampled])
        idx_sorted = np.argsort(X_sampled[:, -1])

        return X_sampled[idx_sorted], None
    else:
        # print(X_all.shape[0], N)
        idx = np.random.choice(X_all.shape[0], N, replace=False)
        X_sampled = X_all[idx, :]
        idx_sorted = np.argsort(X_sampled[:, -1])

        return X_sampled[idx_sorted], idx[idx_sorted] 

def sample_random(X_all, N):
    """Given an array of (x,t) points, sample N points from this."""
    # set_seed(seed) # this can be fixed for all N_f

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]

    return X_sampled, idx 

def sample_random_interval(lb=[0], ub=[1], N=512):
    """Given an array of (x,t) points, sample N points from this."""
    # set_seed(seed) # this can be fixed for all N_f
    point_dim = []
    for lb_i, ub_i in zip(lb, ub):
        point_dim.append(np.random.uniform(low=lb_i, high=ub_i, size=(N,1)))

    return np.hstack(point_dim)

def set_activation(activation):
    if activation == 'identity':
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return  nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        print("WARNING: unknown activation function!")
        return -1

logger_initialized = {}


def init_environ(cfg):
    # build work dir
    exp_name = cfg.name # or config name
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # work_dir = os.path.join('./exp', exp_name, timestamp+'_'+cfg.sub_name)
    work_dir = os.path.join(cfg.exp_dir, exp_name, timestamp+'_'+cfg.sub_name)
    Path(work_dir).mkdir(parents=True, exist_ok=True)  
    cfg.work_dir = work_dir      
    
    # init distributed parallel
    if cfg.launcher == 'slurm':
        _init_dist_slurm('nccl', cfg, cfg.port)
    # else:
    #     raise NotImplementedError(f'launcher {cfg.launcher} has not been implemented.')
    
    # create logger
    log_file = os.path.join(work_dir, 'log.txt')
    logger = get_logger('search', log_file)
    cfg.log_file = log_file
    # set random seed
    # if cfg.seed is not None:
    #     set_random_seed(cfg.seed)
    #     logger.info(f'set random seed to {cfg.seed}')
    
    return logger

        
def _init_dist_slurm(backend, cfg, port=None):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    cfg.world_size = ntasks
    cfg.gpu_id = proc_id % num_gpus
    cfg.rank = proc_id

    dist.init_process_group(backend=backend)
    print(f'Distributed training on {proc_id}/{ntasks}')


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    logger.propagate = False

    return logger


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank = dist.get_rank()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False