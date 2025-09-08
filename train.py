'''Model training.'''
import argparse
import asyncio
import datetime
import multiprocessing
import os
import time

import pandas as pd
import threading
import pickle as pkl
import wandb
import numpy as np
import torch
import torch.nn as nn
import yaml
from pytorch_lamb import Lamb
import common
from torch.utils import data

import datasets
from model import transformer, made
from datasets import JoinOrderBenchmark
#from model.TesseractTransformer import TesseractTransformer
from model.transformer import Transformer
from utils import train_utils, util
import utils
from utils.util import GPUMemoryMonitor

# torch.set_default_dtype(torch.float64)
#
# DEVICE = util.get_device()
# print('Device', DEVICE)
cache_loader = None
cache_ep = 0
global_steps = 1
io_threading_pool = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training.
    # parser.add_argument('--config', type=str, default='DMV-tiny', help='config name')
    parser.add_argument('--config', type=str, default='IMDB', help='config name')
    parser.add_argument('--exp_mark', type=str, default='', help='experiment mark')
    args = parser.parse_args()
    config_name = args.config
    with open(f'./Configs/{args.config}/{args.config}.yaml', 'r', encoding='utf-8') as f:
        global raw_config
        raw_config = yaml.safe_load(f)
    config_seed = raw_config['seed']
    config_excludes = raw_config['excludes']
    config = raw_config['train']



class DataParallelPassthrough(torch.nn.DataParallel):
    """Wraps a model with nn.DataParallel and provides attribute accesses."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret


def TotalGradNorm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def RunEpoch(model,
             opt,
             train_data,
             loader,
             upto=None,
             epoch_num=None,
             epochs=1,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None,
             writer=None,
             constant_lr=None,
             lr_scheduler=None,
             custom_lr_lambda=None,
             label_smoothing=0.,
             warmups=1000,
             job_id=0):
    dataset = train_data
    losses = []

    entropy_gap = []
    global cache_loader
    global cache_ep
    device = util.get_device()
    if config['use_pregen_data'] and os.path.exists('./train_data') and (cache_loader is None or cache_loader.dataset.run_out):
        try:
            tag = util.get_dataset_tag(raw_config)
            filename = f'./train_data/{tag}-{dataset.base_table.name}-ep{cache_ep}.npz'
            # st = time.time()
            pregen_data = np.load(filename)
            # print(f'load data cost: {time.time()-st}')
            training_data_val = pregen_data['training_data_val']
            training_data_pred = pregen_data['training_data_pred']
            training_data_target = pregen_data['training_data_target']
            # st = time.time()
            cache_dataset = common.CacheDataset([training_data_val, training_data_pred, training_data_target])
            # print(f'create CacheDataset cost: {time.time()-st}')
            # st = time.time()
            cache_loader = torch.utils.data.DataLoader(cache_dataset,
                                                       batch_size=config['bs'],
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       # persistent_workers=True,
                                                       # prefetch_factor=2,
                                                       shuffle=False,
                                                       drop_last=False)
            # print(f'create CacheLoader cost: {time.time()-st}')
            cache_ep += 1
            print(f'loading pregen data {filename}')
        except Exception as e:
            print(f'failed to load pregen data with exception: {e}')
    if cache_loader is not None and not cache_loader.dataset.run_out:
        used_loader = cache_loader
        print('using cache loader')
    else:
        cache_loader = None
        used_loader = loader
        print('using data generator')
    training_data_val = []
    training_data_pred = []
    training_data_target = []
    gen_costs = 0
    st = time.time()
    for step, (xb, gen_cost) in enumerate(used_loader):
        gen_costs += gen_cost
        global global_steps

        if constant_lr:
            lr = constant_lr
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        elif custom_lr_lambda:
            lr_scheduler = None
            lr = custom_lr_lambda(global_steps)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        elif lr_scheduler is None and warmups is not None:
            t = warmups
            if warmups < 1:  # A ratio.
                t = int(warmups * min(len(used_loader), upto) * epochs)

            d_model = model.embed_size
            lr = (d_model ** -0.5) * min(
                (global_steps ** -.5), global_steps * (t ** -1.5))
            # lr = min(5e-3, lr)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        else:
            # We'll call lr_scheduler.step() below.
            lr = opt.param_groups[0]['lr']

        if upto and step >= upto:
            break

        xb[0] = xb[0].float().to(device)
        xb[1] = xb[1].float().to(device)
        target = xb[2].float().to(device)
        xb = xb[:2]
        if config['use_pregen_data'] and used_loader == loader:
            xb_0 = xb[0].float().cpu().detach().numpy()
            xb_1 = xb[1].float().cpu().detach().numpy()
            xb_2 = target.float().cpu().detach().numpy()
            training_data_val.append(xb_0) # if training_data_val is None else np.concatenate((training_data_val, xb_0), 0)
            training_data_pred.append(xb_1) # if training_data_pred is None else np.concatenate((training_data_pred, xb_1), 0)
            training_data_target.append(xb_2)#  if training_data_target is None else np.concatenate((training_data_target, xb_2), 0)
        # Forward pass, potentially through several orderings.
        xbhat = model(xb) if not isinstance(model, TesseractTransformer) else model(xb[1], xb[0])
        loss = model.nll(xbhat, target, target, train_data, use_weight=config['use_class_weight']).mean()

        losses.append(loss.detach().item())

        entropy_gap.append(np.abs(loss.item() / np.log(2) - table_bits))
        if step % log_every == 0:
            print(
                '{} epoch {} Iter {}, entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.7f} lr'
                .format(train_data.base_table.name, epoch_num, step,
                        loss.item() / np.log(2) - table_bits,
                        loss.item() / np.log(2), table_bits, lr))
        if writer is not None:
            writer.add_scalar('entropy_gap', loss.item() / np.log(2) - table_bits, global_step=epoch_num)
            writer.add_scalar('loss', loss.item(), global_step=epoch_num)

        opt.zero_grad()
        loss.backward()
        # wandb.log({'loss':loss.item(), 'loss_log2': loss.item()/np.log(2), 'entropy_gap':loss.item() / np.log(2) - table_bits, 'lr':lr})
        # l2_grad_norm = TotalGradNorm(model.parameters())

        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

            # loss_bits = loss.item() / np.log(2)
        global_steps += 1
        if verbose:
            print('epoch average loss: %f' % (np.mean(losses)))
    epoch_cost = time.time() - st
    print(f'epoch cost: {epoch_cost}, gen cost: {gen_costs}, ratio: {gen_costs/epoch_cost}')
    if not os.path.exists('./train_data/'):
        os.makedirs('./train_data/')
    if config['use_pregen_data'] and len(training_data_val) > 0:
        thread = threading.Thread(target=save_train_data(train_data.base_table.name, training_data_val, training_data_pred, training_data_target))
        io_threading_pool.append(thread)
        thread.start()
        cache_ep += 1
    if return_losses:
        return losses, entropy_gap
    return np.mean(losses), np.mean(entropy_gap)


def save_train_data(table_name, training_data_val, training_data_pred, training_data_target):
    training_data_val = np.concatenate(training_data_val, 0)
    training_data_pred = np.concatenate(training_data_pred, 0)
    training_data_target = np.concatenate(training_data_target, 0)
    tag = util.get_dataset_tag(raw_config)
    np.savez_compressed(f'./train_data/{tag}-{table_name}-ep{cache_ep}.npz', training_data_val=training_data_val, training_data_pred=training_data_pred, training_data_target=training_data_target)

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)







def train_fun(dataset, seed, exp_mark, job_id, raw_config, gpu_id=None):
    os.makedirs(os.path.dirname(f'./train_log/{exp_mark}/'), exist_ok=True)
    with open(f'./train_log/{exp_mark}/config.yaml', 'w') as f:
        yaml.dump(raw_config, f)
    # wandb.init(project='AQP', id=exp_mark+f'_{job_id}', config=raw_config, save_code=True, job_type='train', tags=[dataset, 'train',], name=dataset+f'_job_id:{job_id}', group=exp_mark)
    JoinOrderBenchmark.LoadTrueBaseCard(raw_config['tag']) # do this within the subprocess to prevent error
    if gpu_id is None:
        utils.util.gpu_id = job_id % raw_config['num_gpu']
    else:
        utils.util.gpu_id = gpu_id
    if '.csv' in dataset:
        dataset = dataset.replace('.csv', '')
    if dataset == 'cast_info':
        raw_config['train']['upto'] = 256
    if dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv', 'D:/PycharmProjects/naru/datasets/{}',
                                 exclude=config_excludes)
    elif dataset == 'dmv':
        table = datasets.LoadDmv(file_path='D:/PycharmProjects/naru/datasets/{}',
                                 exclude=config_excludes)
    elif args.config == 'IMDB':
        table = datasets.LoadImdb(table=dataset, data_dir=raw_config['data_dir'], use_cols=config['use_cols'],
                                  try_load_parsed=False, tag='')
        # table = datasets.LoadDmv(file_path=f'D:/Users/LU/Desktop/{dataset}.csv')
    else:
        raise Exception(f'Wrong Table:{dataset}')
    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(), [2])[0]

    print(table.data.info())
    table_train = table
    train_data = common.TableDataset(table_train, config['bs'], config['expand_factor'])
    if raw_config['factorize']:
        train_data = common.FactorizedTable(train_data, word_size_bits=config['word_size_bits'], factorize_blacklist=raw_config['factorize_blacklist'])

    model = util.MakeModel(table, train_data, config, raw_config, job_id=job_id)
    model.train()
    model.ConvertToEnsemble()
    mb = ReportModel(model)
    if not isinstance(model, transformer.Transformer):
        print('applying train_utils.weight_init()')
        model.apply(train_utils.weight_init)
    if config['use_data_parallel']:
        model = DataParallelPassthrough(model)

    if config['model_type'] in ['Transformer', 'TesseractTransformer']:
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            # betas=(0.9, 0.98),  # B in Lingvo; in Trfmr paper.
            betas=(0.9, 0.997),  # A in Lingvo.
            eps=1e-9,
        )
    else:
        if config['optimizer'] == 'adam':
            opt = torch.optim.Adam(list(model.parameters()), 2e-4)
        elif config['optimizer'] == 'sgd':
            opt = torch.optim.SGD(list(model.parameters()), 2e-4)
        elif config['optimizer'] == 'lamb':
            opt = Lamb(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(.9, .999), adam=False)
        else:
            print('Using Adagrad')
            opt = torch.optim.Adagrad(list(model.parameters()), 2e-4)


    if raw_config['tag']!='':
        # 前面先初始化模型这里再替换未insert的旧数据用来训练，无论何种情况模型的大小、编码始终以完整数据集为准，避免在旧数据训练+新数据测试时报错
        # eval-IMDB.py测试代码无需处理，因为其不依赖表格数据内容，仅依赖测试集查询和基数文件，生成对应测试集并正确读取即可
        tag=raw_config['tag']
        with open(f'./datasets/job/{dataset}{tag}_index.pkl', 'rb') as f:
            sample_index = pkl.load(f)
        if isinstance(train_data, common.FactorizedTable):
            sample_index_tensor = torch.as_tensor(sample_index).long()
            train_data.factorized_tuples = train_data.factorized_tuples[sample_index_tensor]
            train_data.factorized_tuples_np = train_data.factorized_tuples_np[sample_index]
            train_data.table_dataset.tuples = train_data.table_dataset.tuples[sample_index_tensor]
            train_data.table_dataset.tuples_np = train_data.table_dataset.tuples_np[sample_index]
        train_data.base_table.data = train_data.base_table.data.iloc[sample_index]

    loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=config['bs'],
                                         # sampler=sampler,
                                         shuffle=True,
                                         collate_fn=train_data.collect_fn)


    total_steps = config['epochs'] * len(loader) if config['max_steps'] is None else config['epochs'] * min(config['max_steps'], len(loader))
    print(f'total steps:{total_steps}')
    lr_scheduler = config['lr_scheduler']
    custom_lr_lambda = None
    if lr_scheduler == 'CosineAnnealingLR':
        # Starts decaying to 0 immediately.
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, total_steps)
    elif lr_scheduler == 'OneCycleLR':
        # Warms up to max_lr, then decays to ~0.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=2e-3, total_steps=total_steps)
    elif lr_scheduler is not None and lr_scheduler.startswith(
            'OneCycleLR-'):
        warmup_percentage = float(lr_scheduler.split('-')[-1])
        # Warms up to max_lr, then decays to ~0.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=1e-2,
            total_steps=total_steps,
            pct_start=warmup_percentage)
    elif lr_scheduler is not None and lr_scheduler.startswith(
            'wd_'):
        # Warmups and decays.
        splits = lr_scheduler.split('_')
        assert len(splits) == 3, splits
        lr, warmup_fraction = float(splits[1]), float(splits[2])
        custom_lr_lambda = train_utils.get_cosine_learning_rate_fn(
            total_steps,
            learning_rate=lr,
            min_learning_rate_mult=1e-5,
            constant_fraction=0.,
            warmup_fraction=warmup_fraction)
    else:
        assert lr_scheduler is None, lr_scheduler

    bs = config['bs']
    log_every = 200

    # writer = SummaryWriter(logdir=f'./tbx_log/')
    writer = None
    # os.makedirs(os.path.dirname('./models/'), exist_ok=True)
    train_losses = []
    min_gap = float('inf')
    no_improve_epoch_num = 0
    slow_improve_epoch_num = 0
    last_gap = float('inf')
    if config['patient'] is not None and config['decay_patient'] is not None:
        assert config['patient'] > config['decay_patient']
    cool_down = 0
    decay_times = 0
    constant_lr = config['constant_lr']
    # wandb.log({'table_bits':table_bits})
    # wandb.watch(model, model.nll, log='all', idx=job_id, log_freq=config['max_steps'] if args.config == 'IMDB' else 1000)
    torch.set_grad_enabled(True)
    train_start = time.time()
    log = []
    monitor = GPUMemoryMonitor(interval=0.5)  # 创建显存监控器
    monitor.start()  # 启动监控
    for epoch in range(config['epochs']):
        print(epoch)
        epoch_start = time.time()
        mean_epoch_train_loss, entropy_gap = RunEpoch(model,
                                                      opt,
                                                      train_data=train_data,
                                                      loader=loader,
                                                      upto=config['max_steps'] if args.config == 'IMDB' else None,
                                                      epoch_num=epoch,
                                                      epochs=config['epochs'],
                                                      warmups=config['warmups'],
                                                      constant_lr=constant_lr,
                                                      lr_scheduler=lr_scheduler,
                                                      custom_lr_lambda=custom_lr_lambda,
                                                      log_every=log_every,
                                                      table_bits=table_bits,
                                                      writer=writer,
                                                      job_id=job_id,
                                                      label_smoothing=config['label_smoothing'])
        current_time = time.time()
        cost = current_time-epoch_start
        log.append((epoch, cost, log[-1][2] + cost if len(log)>0 else cost, mean_epoch_train_loss, entropy_gap))
        # if config['num_orders_to_forward'] != 1:
        #     entropy_gap = mean_epoch_train_loss
        if epoch % 1 == 0:
            print('Epoch {} train loss {:.4f} nats / {:.4f} bits, current gap {:.4f}, min gap {:.4f}'.format(
                epoch, mean_epoch_train_loss,
                mean_epoch_train_loss / np.log(2),
                entropy_gap,
                min_gap))
            since_start = time.time() - train_start
            print('time since start: {:.1f} secs'.format(since_start))

        train_losses.append(mean_epoch_train_loss)
        if seed is not None:
            PATH = './Configs/{}/model/{}/{}-seed{}-{}.pt'.format(args.config, exp_mark,dataset, seed, epoch)
        else:
            PATH = './Configs/{}/model/{}/{}-{}.pt'.format(args.config,exp_mark, dataset, epoch)
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        torch.save(model.state_dict(), PATH)
        print(f'{dataset} Saved with entropy_gap:{entropy_gap} min_gap:{min_gap} to:')
        print(PATH)
        if last_gap - entropy_gap > config['slow_improvement_threshold']:
            slow_improve_epoch_num = 0
        else:
            slow_improve_epoch_num += 1
        last_gap = entropy_gap
        if min_gap - entropy_gap > config['no_improvement_threshold']:
            if seed is not None:
                PATH = './Configs/{}/model/{}/{}-seed{}-best.pt'.format(args.config,exp_mark, dataset, seed)
            else:
                PATH = './Configs/{}/model/{}/{}-best.pt'.format( args.config,exp_mark, dataset)

            no_improve_epoch_num = 0
            slow_improve_epoch_num = 0
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            torch.save(model.state_dict(), PATH)
            print(f'{dataset} Saved best with entropy_gap:{entropy_gap} min_gap:{min_gap} to:')
            print(PATH)
            min_gap = entropy_gap
        else:
            no_improve_epoch_num += 1
            slow_improve_epoch_num = 0
        print(f'{dataset} no improvement: {no_improve_epoch_num}, slow improvement: {slow_improve_epoch_num}')
        cool_down += 1
        if config['use_adaptive_scheduler']:
            # assert constant_lr is not None
            if config['decay_patient'] is not None and no_improve_epoch_num >= config['decay_patient'] and cool_down > config['cool_down']:
                # need decay lr
                if config['anneal_patient'] is None or decay_times < config['anneal_patient']:
                    new_lr = constant_lr*config['decay_rate']
                    min_lr = config['min_lr']
                    if new_lr>=min_lr:
                        print(f'{dataset} lr decay from {constant_lr} to {new_lr}')
                        constant_lr = new_lr
                        cool_down = 0
                        decay_times += 1
                    else:
                        print(f'early_stop on: {train_data.base_table.name} due to new_lr: {new_lr}down break min_lr: {min_lr}')
                        break
                else:
                    new_lr = min(constant_lr * np.power(config['decay_rate'], decay_times+1), 1e-3)
                    print(f'{dataset} lr anneal from {constant_lr} to {new_lr}')
                    constant_lr = new_lr
                    cool_down, decay_times = 0, 0
                print(f'{dataset} lr :{constant_lr}')
            if config['patient'] is not None and cool_down > config['cool_down'] and (no_improve_epoch_num > config['patient'] or slow_improve_epoch_num > config['patient']):
                print(f'early_stop on: {train_data.base_table.name}')
                break

    # 停止监控并获取显存峰值
    max_memory_usage = monitor.stop()
    print(f"表 {table.base_table.name} 训练完成，最大显存占用: {max_memory_usage:.4f} GB")
    df = pd.DataFrame(log, columns=['epoch', 'epoch_cost', 'seconds_since_start', 'mean_loss', 'entropy_gap'])

    df['max_gpu_memory_used'] = max_memory_usage
    df.to_csv(f'./train_log/{exp_mark}/{dataset}.csv', index=False)
    for thread in io_threading_pool:
        thread.join()
    if writer is not None:
        writer.close()
    print('Training done; evaluating likelihood on full data:')


class ParallelTrain(multiprocessing.Process):
    def __init__(self, args):
        super().__init__()
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None
        self.args = args

    def run(self):
        #try:
        train_fun(*self.args)
        # except Exception as e:
        #     tb = traceback.format_exc()
        #     print(tb)
        #     self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def TrainTask(seed=0):
    # wandb.login()
    torch.manual_seed(0)
    np.random.seed(0)
    if args.config == 'IMDB':
        dataset_names = list(JoinOrderBenchmark.GetJobLightJoinKeys().keys())
    else:
        dataset_names = raw_config['dataset']
    if args.exp_mark == '':
        exp_mark = str(datetime.datetime.timestamp(datetime.datetime.now()))
    else:
        exp_mark = args.exp_mark
    # dataset_names = ['title']
    print(f'exp_mark: {exp_mark}')
    jobs = []
    # jobs = multiprocessing.Pool(config['multiprocess_pool_size'])
    job_id = 0
    all_gpu = set([g for g in range(raw_config['num_gpu'])]).difference(set(raw_config['exclude_gpus']))
    used_gpu = set()
    launched_table = set()
    table_id = 0
    while len(launched_table) < len(dataset_names):
        if len(jobs) >= config['multiprocess_pool_size'] and len(used_gpu) >= raw_config['num_gpu']:
            for i in range(len(jobs) - 1, -1, -1):
                job = jobs[i]
                if not job.is_alive():
                    jobs.pop(i)
        else:
            dataset = dataset_names[table_id]
            # train_fun(dataset, seed, exp_mark, job_id)
            if len(used_gpu) < len(all_gpu):
                gpu = list(all_gpu.difference(used_gpu))[0]
                jobs.append(ParallelTrain((dataset, seed, exp_mark, job_id, raw_config, gpu)))
                used_gpu.add(gpu)
                launched_table.add(dataset)
            else:
                while job_id%raw_config['num_gpu'] in raw_config['exclude_gpus']:
                    job_id +=1
                jobs.append(ParallelTrain((dataset, seed, exp_mark, job_id, raw_config)))
                used_gpu.add(job_id % raw_config['num_gpu'])
                launched_table.add(dataset)
            print(f'launch: {dataset}, id: {job_id}')
            jobs[-1].start()
            job_id += 1
            table_id += 1
    for job in jobs:
        job.join()
    # wandb.finish()
    # train_fun(dataset, seed, exp_mark)
    # jobs.close()

    # jobs.join()
    # for p in jobs:
    #     p.join()
    # for p in jobs:
    #     if p.exception:
    #         error, traceback = p.exception
    #         print(traceback)


if __name__ == '__main__':
    TrainTask(seed=config_seed)
