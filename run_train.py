import datetime

from utils.train_utils import train_background
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--exp_mark', type=str, default='')
    args = args.parse_args()
    if args.exp_mark == '':
        exp_mark = str(datetime.datetime.timestamp(datetime.datetime.now()))
    else:
        exp_mark = args.exp_mark
    print(f'exp_mark:{exp_mark}')
    train_background(exp_mark)