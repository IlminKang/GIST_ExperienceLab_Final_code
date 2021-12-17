import os
import random
import time
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from utils import seed_rand, TrainQProto, TrainProto, EvalQProto, EvalProto
from dataset import MiniImageNet, CategoriesSampler, IMAGE_PATH, SPLIT_PATH
from models import ProtoNet



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='miniImageNet')
    parser.add_argument('--data_dir', type=str, default='/home/t1_u1/ilmin/workspace/datas')
    # qproto, proto
    parser.add_argument('--model', type=str, default='qproto')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='ResNet')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--weight_path', type=str, default='./models/resnet50/resnet50_cifar100_best_61.pt')
    parser.add_argument('--eval_weight_path', type=str, default='./models/qproto/100_5_way_1_shot_pretrain_True_best_61.pt')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--evaluation_episodes', type=int, default=1000)
    parser.add_argument('--eval', type=bool, default=False)

    args = parser.parse_args()
    print(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    print(f'Using {device}...')

    seed_rand(args.seed, use_cuda)
    if not args.eval:
        print('training')
        print('loading dataset...')
        trainset = MiniImageNet('train', args)
        train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.num_workers)

        valset = MiniImageNet('val', args)
        val_sampler = CategoriesSampler(valset.label, 100, args.way, args.shot + args.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=args.num_workers)
        print(f'creating {args.model} network...')
        model = ProtoNet(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

        model = ProtoNet(args)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

        model.to(device)
        if args.model == 'qproto':
            model = TrainQProto(args, model, optimizer, scheduler, device, train_loader, val_loader)
        elif args.model == 'proto':
            model = TrainProto(args, model, optimizer, scheduler, device, train_loader, val_loader)
        print('Train fin')
    else:
        print('evaluation')
        print('loading dataset...')
        testset = MiniImageNet('test', args)
        test_sampler = CategoriesSampler(testset.label, 100, args.way, args.shot + args.query)
        test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=args.num_workers)
        print(f'loading {args.model} weights...')
        model = ProtoNet(args)
        weight_path = args.eval_weight_path
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        if args.model == 'qproto':
            EvalQProto(args, model, device, test_loader)
        elif args.model == 'proto':
            EvalProto(args, model, device, test_loader)
        print('Eval fin')


