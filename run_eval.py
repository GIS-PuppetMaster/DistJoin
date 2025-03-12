import os

from utils.train_utils import eval_background
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--exp_mark', type=str, default='', help='experiment mark')
# args = parser.parse_args()
if __name__ == "__main__":
    print(f'pid:{os.getpid()}')
    exp_mark = input('input exp mark:')
    eval_background(exp_mark)
