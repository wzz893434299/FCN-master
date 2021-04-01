import torch

from utils import get_config, get_log_dir, get_cuda
from data_loader import get_loader
from trainer import Trainer
import warnings
import argparse

warnings.filterwarnings('ignore')

resume = ''

CALCULATE_DEVICE = 'npu:1'

parser = argparse.ArgumentParser()

# Model hyper-parameters
parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'trainval', 'demo'])
parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--backbone", type=str, default='vgg')
parser.add_argument("--root_dataset", type=str, default='../../home/data/Pascal_VOC/')
parser.add_argument("--resume", type=str, default='')
parser.add_argument("--fcn", type=str, default='32s', choices=['32s', '16s', '8s', '50', '101'])
parser.add_argument('--npu', default=1, type=int, help='NPU id to use.')


def main():
    opts = parser.parse_args()

    if opts.npu is None:
        opts.npu = 0
    global CALCULATE_DEVICE
    # CALCULATE_DEVICE = "npu:{}".format(opts.npu)
    torch.npu.set_device(CALCULATE_DEVICE)
    opts.cuda = torch.device(CALCULATE_DEVICE)
    print("use ", CALCULATE_DEVICE)

    # os.environ['npu_VISIBLE_DEVICES'] = str(opts.gpu_id)

    # opts.npu= get_npu(torch.npu.is_available() and opts.gpu_id != -1,opts.gpu_id)



    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.mode in ['train', 'trainval']:
        opts.out = get_log_dir('fcn' + opts.fcn, 1, cfg)
        print('Output logs: ', opts.out)

    data = get_loader(opts)

    trainer = Trainer(data, opts)
    if opts.mode == 'val':
        trainer.Test()
    elif opts.mode == 'demo':
        trainer.Demo()
    else:
        trainer.Train()


if __name__ == "__main__":
    main()
