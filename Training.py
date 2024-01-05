from random import shuffle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import tqdm
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset import get_loader
import math
from Models.DATFormer import DATFormer
import os


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


def main(local_rank, num_gpus, args):

    torch.set_num_threads(3)
    
    cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = DATFormer(args)
    
    net.train()
    net.cuda()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr},])
    
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                            #    shuffle=True
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
            Methods:{}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset), args.methods))

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    min_loss = 0
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in tqdm.tqdm(enumerate(train_loader)):
            if (i + 1) > iter_num: break

            images, label_224, contour_224 = data_batch

            images, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True)),  \
                                        Variable(contour_224.cuda(local_rank, non_blocking=True))
    
            mask_1_1, cont_1_1 = net(images)

            # saliency loss
            loss1 = criterion(mask_1_1, label_224)
            # contour loss
            c_loss1 = criterion(cont_1_1, contour_224)

            img_total_loss = loss1
            contour_total_loss = c_loss1

            total_loss = img_total_loss + contour_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')
                torch.save(net.state_dict(), args.save_model_dir + args.trainset + '_'  + args.methods  + str(whole_iter_num) + '.pth')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        if epoch == 0:
            min_loss = epoch_total_loss / iter_num
        if epoch_total_loss / iter_num <= min_loss:
            min_loss = epoch_total_loss / iter_num
            torch.save(net.state_dict(), args.save_model_dir + args.trainset + '_'  + args.methods + str(args.lr) + '.pth')







