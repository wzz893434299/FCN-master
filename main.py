import torch.npu

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
    args = parser.parse_args()

    if args.npu is None:
        args.npu = 0
    global CALCULATE_DEVICE
    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    torch.npu.set_device(CALCULATE_DEVICE)
    args.npu = torch.device(CALCULATE_DEVICE)
    print("use ", CALCULATE_DEVICE)

    # os.environ['npu_VISIBLE_DEVICES'] = str(args.gpu_id)

    # args.npu= get_npu(torch.npu.is_available() and args.gpu_id != -1,args.gpu_id)



    cfg = get_config()[1]
    args.cfg = cfg

    if args.mode in ['train', 'trainval']:
        args.out = get_log_dir('fcn' + args.fcn, 1, cfg)
        print('Output logs: ', args.out)

    data = get_loader(args)

    trainer = Trainer(data, args)
    if args.mode == 'val':
        trainer.Test()
    elif args.mode == 'demo':
        trainer.Demo()
    else:
        trainer.Train()


if __name__ == "__main__":
    main()
