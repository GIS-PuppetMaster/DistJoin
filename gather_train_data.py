'''Model training.'''
import argparse
import datetime
import multiprocessing
import os
import wandb
import numpy as np
import torch
import yaml
from tqdm import tqdm
import common
import datasets
from datasets import JoinOrderBenchmark
from utils import util
import utils

# torch.set_default_dtype(torch.float64)
#
# DEVICE = util.get_device()
# print('Device', DEVICE)


parser = argparse.ArgumentParser()

# Training.
# parser.add_argument('--config', type=str, default='DMV-tiny', help='config name')
parser.add_argument('--config', type=str, default='IMDB', help='config name')
parser.add_argument('--exp_mark', type=str, default='', help='experiment mark')
args = parser.parse_args()
config_name = args.config
with open(f'./Configs/{args.config}/{args.config}.yaml', 'r', encoding='utf-8') as f:
    raw_config = yaml.safe_load(f)
config_seed = raw_config['seed']
config_excludes = raw_config['excludes']
config = raw_config['train']


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


def RunEpoch(loader,
             upto=None
             ):
    training_data_val = None
    training_data_pred = None
    training_data_target = None
    for step, xb in enumerate(loader):
        if upto and step >= upto:
            break
        xb[0] = xb[0].float().cpu().detach().numpy()
        xb[1] = xb[1].float().cpu().detach().numpy()
        target = xb[2].float().cpu().detach().numpy()
        training_data_val = xb[0] if training_data_val is None else np.concatenate((training_data_val, xb[0]), 0)
        training_data_pred = xb[1] if training_data_pred is None else np.concatenate((training_data_pred, xb[1]), 0)
        training_data_target = target if training_data_target is None else np.concatenate((training_data_target, target), 0)
    return training_data_val, training_data_pred,training_data_target

def collect_fun(dataset, seed, exp_mark, job_id, gpu_id=None):
    if gpu_id is None:
        utils.util.gpu_id = job_id % raw_config['num_gpu']
    else:
        utils.util.gpu_id = gpu_id
    if '.csv' in dataset:
        dataset = dataset.replace('.csv', '')
    if dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv', 'D:/PycharmProjects/naru/datasets/{}',
                                 exclude=config_excludes)
    elif dataset == 'dmv':
        table = datasets.LoadDmv(file_path='D:/PycharmProjects/naru/datasets/{}',
                                 exclude=config_excludes)
    elif args.config == 'IMDB':
        table = datasets.LoadImdb(table=dataset, data_dir=raw_config['data_dir'], use_cols=config['use_cols'],
                                  try_load_parsed=False)
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

    loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=config['bs'],
                                         shuffle=True,
                                         collate_fn=train_data.collect_fn)

    for epoch in tqdm(range(config['epochs'])):
        print(epoch)
        training_data_val,training_data_pred,training_data_target = RunEpoch(loader, upto=config['max_steps'] if args.config == 'IMDB' else None)
        if not os.path.exists('./train_data/'):
            os.makedirs('./train_data/')
        tag = util.get_dataset_tag(raw_config)
        np.savez_compressed(f'./train_data/{tag}-{dataset}-ep{epoch}.npz', training_data_val=training_data_val, training_data_pred=training_data_pred, training_data_target=training_data_target)


class ParallelTrain(multiprocessing.Process):
    def __init__(self, args):
        super().__init__()
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None
        self.args = args

    def run(self):
        # try:
        collect_fun(*self.args)
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
    # dataset_names = ['movie_info_idx']
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
                jobs.append(ParallelTrain((dataset, seed, exp_mark, job_id, gpu)))
                used_gpu.add(gpu)
                launched_table.add(dataset)
            else:
                while job_id % raw_config['num_gpu'] in raw_config['exclude_gpus']:
                    job_id += 1
                jobs.append(ParallelTrain((dataset, seed, exp_mark, job_id)))
                used_gpu.add(job_id % raw_config['num_gpu'])
                launched_table.add(dataset)
            print(f'launch: {dataset}, id: {job_id}')
            jobs[-1].start()
            job_id += 1
            table_id += 1
    for job in jobs:
        job.join()

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
