import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler
)

from networks.net_factory import net_factory
from utils import losses, ramps, val_2d
from utils.Generate_Prototype import *

# =========================
# DEVICE SETUP
# =========================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument(
    '--root_path',
    type=str,
    default='../data/acdc'
)

parser.add_argument(
    '--exp',
    type=str,
    default='ALHVR'
)

parser.add_argument(
    '--model',
    type=str,
    default='unet_fea_aux'
)

# SMALL SMOKE TEST
parser.add_argument(
    '--max_iterations',
    type=int,
    default=10
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=2
)

parser.add_argument(
    '--labeled_bs',
    type=int,
    default=1
)

parser.add_argument(
    '--deterministic',
    type=int,
    default=1
)

parser.add_argument(
    '--base_lr',
    type=float,
    default=0.01
)

parser.add_argument(
    '--patch_size',
    type=list,
    default=[256, 256]
)

parser.add_argument(
    '--seed',
    type=int,
    default=1337
)

parser.add_argument(
    '--num_classes',
    type=int,
    default=4
)

parser.add_argument(
    '--labelnum',
    type=int,
    default=7
)

parser.add_argument(
    '--consistency',
    type=float,
    default=0.1
)

parser.add_argument(
    '--consistency_rampup',
    type=float,
    default=200.0
)

parser.add_argument(
    '--temperature',
    type=float,
    default=0.1
)

parser.add_argument(
    '--lamda',
    type=float,
    default=1
)

parser.add_argument(
    '--proportion',
    type=float,
    default=0.8
)

parser.add_argument(
    '--scaler',
    type=float,
    default=1
)

args = parser.parse_args()

# =========================
# HELPERS
# =========================
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(
        epoch,
        args.consistency_rampup
    )

def generate_threshold(con, proportion):
    k = int(con.numel() * proportion)
    lowestk_con, _ = torch.topk(
        con.view(-1),
        k,
        largest=False
    )
    return torch.mean(lowestk_con)

def sharpening(P):
    T = 1 / args.temperature
    return P ** T / (P ** T + (1 - P) ** T)

def patients_to_slices(dataset, patiens_num):

    if "ACDC" in dataset or "acdc" in dataset:
        ref_dict = {
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "70": 1312
        }
    else:
        raise ValueError("Dataset not supported")

    return ref_dict[str(patiens_num)]

# =========================
# TRAIN
# =========================
def train(args, snapshot_path):

    num_classes = args.num_classes
    labeled_bs = args.labeled_bs

    print("Creating models...")

    model1 = net_factory(
        net_type="unet_fea_aux",
        in_chns=1,
        class_num=num_classes
    ).to(device)

    model2 = net_factory(
        net_type="unet_fea_aux",
        in_chns=1,
        class_num=num_classes
    ).to(device)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ])
    )

    db_val = BaseDataSets(
        base_dir=args.root_path,
        split="val"
    )

    total_slices = len(db_train)

    labeled_slice = patients_to_slices(
        args.root_path,
        args.labelnum
    )

    print(
        "Total slices:",
        total_slices,
        "Labeled:",
        labeled_slice
    )

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - labeled_bs
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    optimizer1 = optim.SGD(
        model1.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    optimizer2 = optim.SGD(
        model2.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    dice_loss = losses.DiceLoss(
        n_classes=num_classes
    )

    writer = SummaryWriter(
        snapshot_path + '/log'
    )

    iter_num = 0

    max_epoch = args.max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch))

    print("Starting training...")

    for _ in iterator:

        for _, sampled_batch in enumerate(trainloader):

            volume_batch = sampled_batch['image'].to(device)
            label_batch = sampled_batch['label'].to(device)

            outputs1, _, _, _ = model1(volume_batch)
            outputs2, _, _, _ = model2(volume_batch)

            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)

            loss_dice1 = dice_loss(
                outputs_soft1[:labeled_bs],
                label_batch[:labeled_bs].unsqueeze(1)
            )

            loss_dice2 = dice_loss(
                outputs_soft2[:labeled_bs],
                label_batch[:labeled_bs].unsqueeze(1)
            )

            loss_ce1 = F.cross_entropy(
                outputs1[:labeled_bs],
                label_batch[:labeled_bs].long()
            )

            loss_ce2 = F.cross_entropy(
                outputs2[:labeled_bs],
                label_batch[:labeled_bs].long()
            )

            loss1 = loss_dice1 + loss_ce1
            loss2 = loss_dice2 + loss_ce2

            loss = loss1 + loss2

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num += 1

            print(
                f"Iteration {iter_num} | "
                f"Loss: {loss.item():.4f}"
            )

            logging.info(
                f"Iteration {iter_num} "
                f"Loss {loss.item():.4f}"
            )

            writer.add_scalar(
                'train/loss',
                loss.item(),
                iter_num
            )

            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            break

    writer.close()

    print("Training Finished!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    if args.deterministic:

        cudnn.benchmark = False
        cudnn.deterministic = True

        random.seed(args.seed)
        np.random.seed(args.seed)

        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    snapshot_path = (
        "./model/ACDC_{}_{}_labeled/{}".format(
            args.exp,
            args.labelnum,
            args.model
        )
    )

    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s] %(message)s'
    )

    logging.getLogger().addHandler(
        logging.StreamHandler(sys.stdout)
    )

    logging.info(str(args))

    train(args, snapshot_path)