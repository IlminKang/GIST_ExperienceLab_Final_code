import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Function
import os
import random
import time
import pandas as pd
from tqdm import tqdm



def seed_rand(seed, use_cuda):
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def TrainQProto(args, model, optimizer, scheduler, device, train_loader, val_loader):
    print(f'training {args.model} network...')
    if args.model == 'qproto':
        path = args.save_dir
        path = os.path.join(path, args.model)
        if not os.path.exists(path):
            os.makedirs(path)

        train_log = list()
        val_log = list()
        best_acc = 0
        best_epoch = 0

        for epoch in range(1, args.max_epoch + 1):
            model.train()
            epoch_loss = list()
            epoch_proto_loss = list()
            epoch_vae_loss = list()
            epoch_acc = list()
            start = time.time()

            model.train()

            for idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
                logits, feautures, x_tilde, z_e_x, z_q_x = model(data_shot, data_query)

                query_label = torch.arange(args.way).repeat(args.query).to(device)

                proto_loss = F.cross_entropy(logits, query_label)

                # Reconstruction loss
                loss_recons = F.mse_loss(x_tilde, feautures)
                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                # Commitment objective
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                vae_loss = loss_recons + loss_vq + args.beta * loss_commit

                loss = proto_loss + vae_loss

                acc = count_acc(logits, query_label)

                epoch_loss.append(loss.item())
                epoch_proto_loss.append(proto_loss.item())
                epoch_vae_loss.append(vae_loss.item())
                epoch_acc.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_proto_loss = sum(epoch_proto_loss) / len(epoch_proto_loss)
            epoch_vae_loss = sum(epoch_vae_loss) / len(epoch_vae_loss)
            epoch_acc = sum(epoch_acc) / len(epoch_acc)
            end = time.time()
            epoch_time = end - start
            print(
                f'Train | Epoch ({epoch}/{args.max_epoch}) | Time {epoch_time:.2f}sec | Acc : {epoch_acc:.2f} | Loss : {epoch_loss:.2f} | ProtoLoss : {epoch_proto_loss:.2f} | VAELoss : {epoch_vae_loss:.2f}')
            train_log.append([epoch_loss, epoch_proto_loss, epoch_vae_loss, epoch_acc])

            model.eval()
            epoch_val_loss = list()
            epoch_val_proto_loss = list()
            epoch_val_vae_loss = list()
            epoch_val_acc = list()
            start = time.time()
            with torch.no_grad():
                for idx, (data, label) in enumerate(val_loader):
                    data = data.to(device)
                    p = args.shot * args.way
                    data_shot, data_query = data[:p], data[p:]
                    logits, feautures, x_tilde, z_e_x, z_q_x = model(data_shot, data_query)

                    query_label = torch.arange(args.way).repeat(args.query).to(device)

                    proto_loss = F.cross_entropy(logits, query_label)

                    # Reconstruction loss
                    loss_recons = F.mse_loss(x_tilde, feautures)
                    # Vector quantization objective
                    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                    # Commitment objective
                    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                    vae_loss = loss_recons + loss_vq + args.beta * loss_commit

                    loss = proto_loss + vae_loss

                    acc = count_acc(logits, query_label)

                    epoch_val_loss.append(loss.item())
                    epoch_val_proto_loss.append(proto_loss.item())
                    epoch_val_vae_loss.append(vae_loss.item())
                    epoch_val_acc.append(acc)

            epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
            epoch_val_proto_loss = sum(epoch_val_proto_loss) / len(epoch_val_proto_loss)
            epoch_val_vae_loss = sum(epoch_val_vae_loss) / len(epoch_val_vae_loss)
            epoch_val_acc = sum(epoch_val_acc) / len(epoch_val_acc)
            end = time.time()
            epoch_time = end - start
            print(
                f'Valid | Epoch ({epoch}/{args.max_epoch}) | Time {epoch_time:.2f}sec | Acc : {epoch_val_acc:.2f} | Loss : {epoch_val_loss:.2f} | ProtoLoss : {epoch_val_proto_loss:.2f} | VAELoss : {epoch_val_vae_loss:.2f}')

            val_log.append([epoch_val_loss, epoch_val_proto_loss, epoch_val_vae_loss, epoch_val_acc])

            if best_acc < epoch_val_acc:
                remove_path = os.path.join(path,
                                           f'{args.way}_way_{args.shot}_shot_pretrain_{args.pretrained}_best_{best_epoch}.pt')
                if os.path.exists(remove_path):
                    os.remove(remove_path)
                best_acc = epoch_val_acc
                best_epoch = epoch
                save_path = os.path.join(path,
                                         f'{args.way}_way_{args.shot}_shot_pretrain_{args.pretrained}_best_{best_epoch}.pt')
                print(f'Saving model to {save_path}...')
                # torch.save(model.state_dict(), save_path)

            print()

        log = pd.DataFrame({"train_loss": [x[0] for x in train_log],
                            "train_proto_loss": [x[1] for x in train_log],
                            "train_vae_loss": [x[2] for x in train_log],
                            "train_acc": [x[3] for x in train_log],
                            "val_loss": [x[0] for x in val_log],
                            "val_proto_loss": [x[1] for x in val_log],
                            "val_vae_loss": [x[2] for x in val_log],
                            "val_acc": [x[3] for x in val_log]})
        log_path = os.path.join(path, f'{args.way}_way_{args.shot}_shot_pretrain_{args.pretrained}_log.csv')
        # log.to_csv(log_path)
    return model

def TrainProto(args, model, optimizer, scheduler, device, train_loader, val_loader):
    print(f'training {args.model} network...')
    if args.model == 'proto':
        path = args.save_dir
        path = os.path.join(path, args.model)
        if not os.path.exists(path):
            os.makedirs(path)

        train_log = list()
        val_log = list()
        best_acc = 0
        best_epoch = 0

        for epoch in range(1, args.max_epoch + 1):
            model.train()
            epoch_loss = list()
            epoch_acc = list()
            start = time.time()

            model.train()

            for idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
                logits = model(data_shot, data_query)

                query_label = torch.arange(args.way).repeat(args.query).to(device)

                proto_loss = F.cross_entropy(logits, query_label)

                loss = proto_loss

                acc = count_acc(logits, query_label)

                epoch_loss.append(loss.item())
                epoch_acc.append(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_acc = sum(epoch_acc) / len(epoch_acc)
            end = time.time()
            epoch_time = end - start
            print(
                f'Train | Epoch ({epoch}/{args.max_epoch}) | Time {epoch_time:.2f}sec | Acc : {epoch_acc:.2f} | Loss : {epoch_loss:.2f}')
            train_log.append([epoch_loss, epoch_acc])

            model.eval()
            epoch_val_loss = list()
            epoch_val_acc = list()
            start = time.time()
            with torch.no_grad():
                for idx, (data, label) in enumerate(val_loader):
                    data = data.to(device)
                    p = args.shot * args.way
                    data_shot, data_query = data[:p], data[p:]
                    logits = model(data_shot, data_query)

                    query_label = torch.arange(args.way).repeat(args.query).to(device)

                    proto_loss = F.cross_entropy(logits, query_label)

                    loss = proto_loss

                    acc = count_acc(logits, query_label)

                    epoch_val_loss.append(loss.item())
                    epoch_val_acc.append(acc)

            epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
            epoch_val_acc = sum(epoch_val_acc) / len(epoch_val_acc)
            end = time.time()
            epoch_time = end - start
            print(
                f'Valid | Epoch ({epoch}/{args.max_epoch}) | Time {epoch_time:.2f}sec | Acc : {epoch_val_acc:.2f} | Loss : {epoch_val_loss:.2f}')

            val_log.append([epoch_val_loss, epoch_val_acc])

            if best_acc < epoch_val_acc:
                remove_path = os.path.join(path,
                                           f'{args.way}_way_{args.shot}_shot_pretrain_{args.pretrained}_best_{best_epoch}.pt')
                if os.path.exists(remove_path):
                    os.remove(remove_path)
                best_acc = epoch_val_acc
                best_epoch = epoch
                save_path = os.path.join(path,
                                         f'{args.way}_way_{args.shot}_shot_pretrain_{args.pretrained}_best_{best_epoch}.pt')
                print(f'Saving model to {save_path}...')
                # torch.save(model.state_dict(), save_path)

            print()

        log = pd.DataFrame({"train_loss": [x[0] for x in train_log],
                            "train_acc": [x[1] for x in train_log],
                            "val_loss": [x[0] for x in val_log],
                            "val_acc": [x[1] for x in val_log]})
        log_path = os.path.join(path, f'{args.way}_way_{args.shot}_shot_pretrain_{args.pretrained}_log.csv')
        # log.to_csv(log_path)
    return model

def EvalQProto(args, model, device, test_loader):
    model.eval()
    if args.model == 'qproto':
        print(f'Evaluating {args.model}...')
        avg_acc = list()

        for data, label in tqdm(test_loader):
            with torch.no_grad():
                data = data.to(device)
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
                logits, feautures, x_tilde, z_e_x, z_q_x = model(data_shot, data_query)

                query_label = torch.arange(args.way).repeat(args.query).to(device)

                acc = count_acc(logits, query_label)

                avg_acc.append(acc)
        avg_acc = sum(avg_acc) / len(avg_acc)
        print(f'Test | {args.model} | Acc : {avg_acc:.2f}')

def EvalProto(args, model, device, test_loader):
    model.eval()
    if args.model == 'proto':
        print(f'Evaluating {args.model}...')
        avg_acc = list()

        for data, label in tqdm(val_loader):
            with torch.no_grad():
                data = data.to(device)
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
                logits = model(data_shot, data_query)

                query_label = torch.arange(args.way).repeat(args.query).to(device)

                acc = count_acc(logits, query_label)

                avg_acc.append(acc)
        avg_acc = sum(avg_acc) / len(avg_acc)
        print(f'Test | {args.model} | Acc : {avg_acc:.2f}')
