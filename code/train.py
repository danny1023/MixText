import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytransformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from read_data import get_data
from mixtext import MixText


parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='运行多少epoch')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='有标签数据训练批次大小')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='无标签数据训练批次大小')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='bert的初始学习率')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='Mixtext的模型初始学习率')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='划分出有标签数据的数据个数')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='划分出无标签数据的数据个数')

parser.add_argument('--val-iteration', type=int, default=200,
                    help='迭代次数')

parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix选项,是否要进行mix操作')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix方法, 设置不同的mix方法')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='从有标签和无标签数据分开mix')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='训练时是否随机选择在mix和unmix之间')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='训练数据是否数据增强')

parser.add_argument('--model', type=str, default='bert-base-chinese',
                    help='使用的预训练模型,默认中文bert模型')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='数据集路径')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='mix的层的集合，指定那些层做mix')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='beta分布的alpha参数')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='无标签数据连续损失的权重')
parser.add_argument('--T', default=0.5, type=float,
                    help='sharpen function的温度选项')

parser.add_argument('--temp-change', default=1000000, type=int, help='步数大于多少时改变sharpen function的温度值')

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='hinge loss边界')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='无标签数据的hinge loss权重')

args = parser.parse_args()

#GPU相关设置, 设置为0表示不适用gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if n_gpu == 0:
    print("GPU 数量为0，使用CPU")
else:
    print("可以使用的GPU 数量: ", n_gpu)


best_acc = 0
total_steps = 0
flag = 0
print('是否要进行mix: ', args.mix_option)
print("Mix层集合: ", args.mix_layers_set)


def main():
    global best_acc
    # 读取 dataset 和构建 dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug)
    # 制作loader
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    # 定义模型，设置优化器
    if n_gpu == 0:
        model = MixText(n_labels, args.mix_option, model=args.model)
    else:
        model = MixText(n_labels, args.mix_option, model=args.model).cuda()
    model = nn.DataParallel(model)
    #优化器参数
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])
    #预热步数
    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration
    #是否动态更新学习率
    scheduler = None
    #WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)
    #训练损失函数，用在训练时
    train_criterion = SemiLoss()
    #交叉熵损失, 用在验证集和测试集
    criterion = nn.CrossEntropyLoss()
    test_accs = []

    #开始训练
    for epoch in range(args.epochs):
        #调用train函数, 给定有标签数据，无标签数据，模型，优化器，损失函数，labels数量，是否数据增强
        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug)

        # scheduler.step()
        # _, train_acc = validate(labeled_trainloader, model,  criterion, epoch, mode='Train Stats')
        #print("epoch {}, train acc {}".format(epoch, train_acc))

        val_loss, val_acc = validate(val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, 验证集准确率 {}, 验证集损失 {}".format(epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, 测试集准确率 {},测试集损失 {}".format(epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('最好的准确率')
        print(best_acc)

        print('测试集准确率')
        print(test_accs)

    print("完成训练")
    print('最好的准去率')
    print(best_acc)

    print('测试准确率')
    print(test_accs)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    """
    :param labeled_trainloader: 有标签数据
    :param unlabeled_trainloader: 无标签数据
    :param model:  Mixtext 模型
    :param optimizer: 优化器
    :param scheduler: 动态更新学习率
    :param criterion: 损失函数
    :param epoch:
    :param n_labels: 标签类别数量
    :param train_aug: 是否使用数据增强
    :return:
    """
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()
    #设定更改温度T的条件
    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        print('改变sharpen function的温度选项')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1
        #如果训练集不做数据增强
        if not train_aug:
            inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
        #训练集做数据增强
        else:
            (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length, inputs_x_length_aug) = labeled_train_iter.next()
            (inputs_u,  inputs_ori), (length_u,  length_ori) = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)
        #原始输入句子的批次
        batch_size_2 = inputs_ori.size(0)
        #targets_x 做成one_hot编码
        targets_x = torch.zeros(batch_size, n_labels).scatter_(1, targets_x.view(-1, 1), 1)
        if n_gpu != 0:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_ori = inputs_ori.cuda()

        mask = []

        with torch.no_grad():
            # 对无标签数据预测标签, 预测出的类别置信度, 首先猜测无标签数据的低熵标签
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            outputs_ori = model(inputs_ori)

            # 根据不同数据翻译后的质量，给与不同的权重
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (1)
            # sharpen 计算，p的n次方, 如果T是0.5，那么pt就是p的平方值
            pt = p**(1/args.T)
            # targets_u 是求一个百分比
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
        #是否使用mix, 1表示已经mix，是一个flag
        mixed = 1

        # 训练时是否随机选择在mix和unmix之间
        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1
        # 如果使用mix，设置beta分布参数 beta分布的alpha参数 lbeta, lbeta是一个浮点数, beta分布的参数alpha,beta, Dirichlet 分布是由 Beta 分布推广而来的
        if mix_ == 1:
            lbeta = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                lbeta = lbeta
            else:
                lbeta = max(lbeta, 1-lbeta)
        else:
            lbeta = 1
        #选择哪一层进行mix_layer, 随机选择一层，减去1是为了和列表索引相对应,例如选择bert的第11层
        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        #如果不使用数据增强
        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_ori, inputs_ori], dim=0)

            all_lengths = torch.cat(
                [inputs_x_length, length_u, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        #如果使用数据增强，拼接所有输入，长度和标签（无标签数据预测出来的标签)
        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x_aug, inputs_u, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)
        #分别进行mix, 是使用batch_size个还是全部，是从batch_size 获取，还是从全部数据获取
        if args.separate_mix:
            #随机打乱函数randperm，获取索引，就是把无标签，有标签和数据增强的数据随机抽取出来进行训练
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)
        #input_a是所有输入，input_b是抽取的部分输入,input和target都是embedding后的向量，length_a,length_b是原始的seq_length,未加padding时候的
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix句子的隐藏向量,input_a是所有输入，input_b是随机抽取的输入
            logits = model(input_a, input_b, lbeta, mix_layer)
            #对target也使用Mixup混合，公式是论文上的公式
            mixed_target = lbeta * target_a + (1 - lbeta) * target_b
        elif args.mix_method == 1:
            # 拼接2个训练句子的片段，  片段的选择根据beta分布lbeta
            # 例如: "I love you so much" 和 "He likes NLP" 可能混合成 "He likes NLP so much".
            # 对应的labels也会根据系数被混合
            mixed_input = []
            if lbeta != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * lbeta)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1-lbeta))
                    if length1 + length2 > 256:
                        length2 = 256-length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        if n_gpu !=0:
                            mixed_input.append(torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2)).cuda()), dim=0).unsqueeze(0))
                        else:
                            mixed_input.append(torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2))), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits = model(mixed_input)
            mixed_target = lbeta * target_a + (1 - lbeta) * target_b

        elif args.mix_method == 2:
            # 拼接2个句子
            # 对应的label是平均值
            if lbeta == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    if n_gpu != 0:
                        mixed_input.append(torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i]))).cuda()), dim=0).unsqueeze(0))
                    else:
                        mixed_input.append(torch.cat((input_a[i][:length_a[i]], torch.tensor([102]), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i])))), dim=0).unsqueeze(0))
                mixed_input = torch.cat(mixed_input, dim=0)
                logits = model(mixed_input, sent_size=512)

                #mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b)/2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args.val_iteration, mixed)

        if mix_ == 1:
            loss = Lx + w * Lu
        else:
            loss = Lx + w * Lu + w2 * Lu2

        #max_grad_norm = 1.0
        # 梯度裁剪
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            if n_gpu !=0:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):
        """
        半监督损失函数
        :param outputs_x: 模型输出的x
        :param targets_x: 真实的x
        :param outputs_u: 模型输出的无标签的x
        :param targets_u:  真实的无标签的x
        :param outputs_u_2: 模型输出的无标签x_2
        :param epoch:  迭代次数
        :param mixed:
        :return:
        """
        if args.mix_method == 0 or args.mix_method == 1:
            #有监督的x的损失
            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))
            #无监督的x输出的概率值
            probs_u = torch.softmax(outputs_u, dim=1)
            #论文中公式显示的kl散度, batch mean 批次均值作为统计计算KL散度
            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')
            #计算hinge Loss 折页损失 max(0,1-(wTxi +b)yi)
            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
