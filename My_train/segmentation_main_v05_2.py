import datetime
import os
import random
import time
import torchvision.models as models
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import My_train.joint_transforms as joint_transforms
import My_train.transforms as extended_transforms
import My_train.color_transforms as colorjitter
import My_train.misc as misc
import argparse
import torch.nn.functional

from cfgs import DenseASPP121
from cfgs import DenseASPP161
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from My_train import segmentation_dataloader
from models.DenseASPP import *
from My_train.misc import check_mkdir, evaluate, AverageMeter
from collections import OrderedDict


parser = argparse.ArgumentParser(description='DenseASPP training')
parser.add_argument('--input_height',           type=int,   help='input height', default=512)
parser.add_argument('--input_width',            type=int,   help='input width', default=512)
parser.add_argument('--train_batch_size',       type=int,   help='train batch size', default=4)
parser.add_argument('--val_batch_size',         type=int,   help='validation batch size', default=4)
parser.add_argument('--num_threads',            type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate', default=3e-4)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs', default=80)
parser.add_argument('--weight_decay',           type=float, help='weight decay', default=1e-5)
parser.add_argument('--print_frequency',        type=int,   help='print frequency', default=2)
parser.add_argument('--val_save_to_img_file',   type=bool,  help='save validation image file', default=True)
parser.add_argument('--val_img_sample_rate',    type=float, help='randomly sample some validation results to display', default=0.05)
parser.add_argument('--checkpoint_path',        type=str,   help='path ro a specific checkpoint to load',
                    default='/home/mk/Semantic_Segmentation/DenseASPP-master/pretrained_model/densenet121.pth')
parser.add_argument('--retrain',                            help='if used with checkpoint path, will restart training from step zero', action='store_true')

args = parser.parse_args()

x = 5
version = '2_1'

ckpt_path = '../../ckpt'
ImageNet = 'ImageNet/DenseNet121'
exp_name_ImageNet = 'segImageNet_v0{}_{}'.format(x, version)

# [[ SummaryWriter]]
# Writes 'Summary' directly to event files.
# writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))
writer = SummaryWriter(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet))


def poly_lr_scheduler(optimizer, init_lr, epoch, maxEpoch=args.num_epochs, power=0.9):
    "init_lr      : base learning rate \
    iter          : current iteration \
    lr_decay_iter : how frequently decay occurs, default is 1 \
    power         : polynomial power"

    # how get more smaller lr than previously lr
    # epoch 23 ?
    # if 50 <= epoch < 80:.
    #     factor = epoch % 100
    #     lr = init_lr * (1 - (epoch - factor)/ maxEpoch) ** power
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # elif 1 <= epoch < 50:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = init_lr
    # else:
    #     # factor = epoch % 100
    #     lr = init_lr * (1 - epoch / maxEpoch) ** power
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    lr = init_lr * (1 - epoch / maxEpoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    net = DenseASPP(model_cfg=DenseASPP121.Model_CFG).cuda()
    # densenet121 = models.densenet121(pretrained=True)

    if len(args.checkpoint_path) == 0:
        curr_epoch = 1
        # Initializing 'best_record'
        args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        # load the pretrained model
        print('training resumes from ' + args.checkpoint_path)
        # lambda argument: manipulate(argument)
        # pretrained_weight = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        pretrained_weight = torch.load(args.checkpoint_path)
        new_state_dict = OrderedDict()
        model_dict = net.state_dict()
        for key, value in pretrained_weight.items():
            name = key
            # print(name)
            new_state_dict[name] = value

        new_state_dict.pop('features.norm5.weight')
        new_state_dict.pop('features.norm5.bias')
        new_state_dict.pop('features.norm5.running_mean')
        new_state_dict.pop('features.norm5.running_var')
        new_state_dict.pop('classifier.weight')
        new_state_dict.pop('classifier.bias')
        # print(new_state_dict)
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict, strict=False)
        # pretrained_dict = {key: value for key, value in pretrained_dict.items() if key in model_dict}
        # model_dict.update(pretrained_dict)
        # pretrained_dict = {key: value for key, value in pretrained_dict.items() if key != 'classifier.weight' or 'classifier.bias'}
        # for key, value in pretrained_dict.items():
        #     if 'classifier.weight' in key:
        #         key = key.rstrip('classifier.weight')
        #     if 'classifier.bias' in key:
        #         key = key.rstrip('classifier.bias')
        #     if 'features.norm5.weight' in key:
        #         key = key.replace("features.norm5.weight", "")
        #     elif 'features.norm5.bias' in key:
        #         key = key.strip('features.norm5.bias')
        #     elif 'features.norm5.running_mean' in key:
        #         key = key.strip('features.norm5.running_mean')
        #     elif 'features.norm5.running_var' in key:
        #         key = key.strip('features.norm5.running_var')
        #     name = key
        #     print(name)
        #     new_pretrained_dict[name] = value

        # model.load_state_dict(model_dict, strict=False)
        # model.load_state_dict(new_pretrained_dict, strict=False)
        curr_epoch = 1
        args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
        # split_snapshot = args.checkpoint_path.split('_')
        # curr_epoch = int(split_snapshot[1]) + 1
        # args.best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
        #                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
        #                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    cudnn.benchmark = True

    # net.train()
    # tells your model that you are training the model ,or sets the module in training mode.
    # The classic workflow : call train() --> epoch of training on the training set --> call eval()
    # --> evaluate your model on the validation set

    # mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mean_std = ([0.290101, 0.328081, 0.286964], [0.182954, 0.186566, 0.184475])

    # ---------------------------------- [[ data - augmentation ]] ---------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # [[joint_transforms]]
    # both raw image and gt are transformed by data-augmentation
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomSizedCrop(size=args.input_width),
        # joint_transforms.RandomSized(size=args.input_width),
        joint_transforms.RandomHorizontallyFlip()
    ])

    val_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomSizedCrop(size=args.input_width)
    ])

    # transform : To preprocess images
    # Compose : if there are a lot of preprocessed images, compose plays a role as collector in a single space.
    input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(hue=0.1),
        # colorjitter.ColorJitter(brightness=0.1),
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*mean_std)
    ])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(hue=0.1),
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*mean_std)
    ])

    # target_transform = extended_transforms.MaskToTensor()
    target_transform = extended_transforms.Compose([
        extended_transforms.MaskToTensor(),
    ])

    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        # [[ ToPILImage() ]]
        # Convert a tensor or an ndarray to PIL Image.
        standard_transforms.ToPILImage()
    ])
    visualize = standard_transforms.ToTensor()
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    train_set = segmentation_dataloader.CityScapes('fine', 'train', joint_transform=train_joint_transform,
                                                   transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_threads, shuffle=True)

    val_set = segmentation_dataloader.CityScapes('fine', 'val', joint_transform=val_joint_transform,
                                                 transform=val_input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=args.num_threads, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=segmentation_dataloader.ignore_label)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # optimizer = optim.Adam([
    #     {   'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #         'lr': args.learning_rate}
    # ], weight_decay=args.weight_decay)

    # if len(args.checkpoint_path) > 0:
    #     optimizer.load_state_dict(torch.load(args.checkpoint_path))
    #     optimizer.param_groups[0]['lr'] = 2 * args.learning_rate
    #     optimizer.param_groups[1]['lr'] = args.learning_rate

    check_mkdir(ckpt_path)
    # check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet))
    # open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')
    open(os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')
    # lambda1 = lambda epoch : (1 - epoch // args['epoch_num']) ** 0.9

    # [[learning-rate decay]]
    # factor = (1 - curr_epoch/ args['epoch_num'])**0.9
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor= factor, patience=args['lr_patience'],
    #                                                  min_lr=0)

    for epoch in range(curr_epoch, args.num_epochs + 1):
        # factor = (1 - epoch / args['epoch_num']) ** 0.9
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=args['lr_patience'],
        #                                                  min_lr=0)

        # [[ training ]]
        train(train_loader, net, criterion, optimizer, epoch, args, train_set)
        # train(train_loader, net, optimizer, epoch, args, train_set)
        # [[ validation ]]
        validate(val_loader, net, criterion, optimizer, epoch, args, restore_transform, visualize)
        # validate(val_loader, net, optimizer, epoch, args, restore_transform, visualize)
        # scheduler.step(val_loss)

    print('Training Done!!')

def train(train_loader, net, criterion, optimizer, epoch, train_args, train_set):
    import shutil
    src = "/home/mk/Semantic_Segmentation/DenseASPP-master/My_train/segmentation_main2.py"
    copy_path = os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, "segmentation_main2_" +"v_0{}_{}.py".format(x, version))
    shutil.copy(src, copy_path)

    net.train()
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    examples_time = AverageMeter()

    num_training_samples = len(train_set)
    steps_per_epoch = np.ceil(num_training_samples / args.train_batch_size).astype(np.int32)
    num_total_steps = args.num_epochs * steps_per_epoch

    print("total number of samples: {}".format(num_training_samples))
    print("total number of steps  : {}".format(num_total_steps))

    # curr_iter : total dataset per epoch
    curr_iter = (epoch - 1) * len(train_loader)

    # COUNT_PARAMS
    total_num_paramters = 0
    for param in net.parameters():
        total_num_paramters += np.array(list(param.size())).prod()

    print("number of trainable parameters: {}".format(total_num_paramters))
    # for step in range(num_total_steps):
    #     if step and step % 100 == 0:
    #         time_sofar = (time.time() - start_time) / 3600
    #         training_time_left = (num_total_steps / step - 1.0) * time_sofar
    # Data = [[train_loader], [range(num_total_steps)]]
    # for [[i, data], step] in Data :
    index = 0
    start_time = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == segmentation_dataloader.num_classes

        before_op_time = time.time()
        # loss = torch.nn.functional.cross_entropy(input=outputs, target=labels, ignore_index=segmentation_dataloader.ignore_label)
        loss = criterion(outputs, labels)
        duration = time.time() - before_op_time

        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - start_time)

        # why use N?? N is batch size?
        train_loss.update(loss.data[0], N)
        curr_iter += 1

        # [[ writer.add_scalar ]]
        # writer.add_scalar('myscalar', value, iteration)
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args.print_frequency == 0:
            examples_time.update(args.train_batch_size / duration)
            # print_string = 'epoch {: %d} | iter { %d / %d} | train_loss: {%.5f} | time_elapsed: {%.2f}h'
            # print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
            # print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
            print('epoch: %d | iter: %d / %d | train loss: %.5f | examples/s: %4.2f | time_elapsed: %.5f''s' %
                  (epoch, i + 1, len(train_loader), train_loss.avg, examples_time.avg, batch_time.avg))

        poly_lr_scheduler(optimizer=optimizer, init_lr=args.learning_rate, epoch=epoch-1)
        # misc.PolyLR(optimizer=optimizer, curr_iter=epoch-1, max_iter=args.num_epochs, lr_decay=0.9)
        with open(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'LR_v0{}_{}.txt'.format(x,version)), 'a') as LRtxt:
            LRtxt.write("index : {}, epoch : {}, learning rate : {: f}".format(index, epoch, optimizer.param_groups[0]['lr']) + '\n')
            index += 1

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()
    val_loss = AverageMeter()

    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()

        outputs = net(inputs)
        # outputs.data : when batch is 0, pixel value in 19 classes
        predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data[0] / N, N)
        # validation_loss = torch.nn.functional.cross_entropy(input=outputs, target=gts, ignore_index=segmentation_dataloader.ignore_label)
        # val_loss.update(validation_loss.data[0] / N, N)
        for i in inputs:
            if random.random() > train_args.val_img_sample_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(i.data.cpu())
        gts_all.append(gts.data.cpu().numpy())
        predictions_all.append(predictions)

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)

    acc, acc_cls, acc_cls_mean, mean_iu, fwavacc = evaluate(predictions_all, gts_all, segmentation_dataloader.num_classes)

    num_validate = epoch
    with open(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'class_accuracy{}_{}.txt'.format(x, version)), 'a') as acc_cls_txt:
        acc_cls_txt.write("================================the number of validation : {}================================"
                          "\nroad: {}, \nsidewalk: {}, \nbuilding: {}, \nwall: {}, \nfence: {}, \npole: {}, \ntraffic light: {}, \ntraffic sign: {},"
                          "\nvegetation: {}, \nterrain: {}, \nsky: {}, \nperson: {}, \nrider: {}, \ncar: {}, \ntruck: {}, \nbus: {}, \ntrain: {}, \nmotorcycle: {},"
                          "\nbicycle: {}\n\n".format(num_validate, acc_cls[0] * 100, acc_cls[1] * 100, acc_cls[2] * 100,
                                                     acc_cls[3] * 100, acc_cls[4] * 100,
                                                     acc_cls[5] * 100, acc_cls[6] * 100, acc_cls[7] * 100, acc_cls[8] * 100,
                                                     acc_cls[9] * 100, acc_cls[10] * 100, acc_cls[11] * 100, acc_cls[12] * 100,
                                                     acc_cls[13] * 100, acc_cls[14] * 100, acc_cls[15] * 100,
                                                     acc_cls[16] * 100, acc_cls[17] * 100, acc_cls[18] * 100))

    if mean_iu > train_args.best_record['mean_iu']:
        train_args.best_record['val_loss'] = val_loss.avg
        train_args.best_record['epoch'] = epoch
        train_args.best_record['acc'] = acc
        # acc_cls : accuracy class
        train_args.best_record['acc_cls_mean'] = acc_cls_mean
        # mean_iu : mean_intersection over union
        train_args.best_record['mean_iu'] = mean_iu
        # fwavacc : frequency weighted average accuracy
        train_args.best_record['fwavacc'] = fwavacc
        # snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
        #     epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[0]['lr']
        # )
        snapshot_name = 'epoch_%d_loss_%.2f_acc_%.2f_acc-cls_%.2f_mean-iu_%.2f_fwavacc_%.2f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls_mean, mean_iu, fwavacc, optimizer.param_groups[0]['lr']
        )
        torch.save(net.state_dict(), os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet, snapshot_name + '_v0{}'.format(x) + '.pth'))
        # torch.save(optimizer.state_dict(),os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet, 'opt_' + snapshot_name + '_v0{}'.format(x) + '.pth'))

        # setting path to save the val_img
        if train_args.val_save_to_img_file:
            # to_save_dir = os.path.join(ckpt_path, exp_name, 'epoch'+str(epoch)+'_v0{}'.format(x))
            to_save_dir = os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'epoch' + str(epoch) + '_v0{}'.format(x))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue

            # data[0] : inputs_all
            input_pil = restore(data[0])
            gt_pil = segmentation_dataloader.colorize_mask(data[1])
            predictions_pil = segmentation_dataloader.colorize_mask(data[2])

            if train_args.val_save_to_img_file:
                # saving the restored image
                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))

            # input RGB image, gt image and prediction image are showed on tensorboardX
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
        val_visual = torch.stack(val_visual, 0)

        # [[ make_grid() ]]
        # make_grid function : prepare the image array and send the result to add_image()
        # --------------------- make_grid takes a 4D tensor and returns tiled images in 3D tensor ---------------------
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=0)

        # [[ writer.add_image ]]
        # writer.add_image('imresult', x, iteration) : save the image.
        writer.add_image(snapshot_name, val_visual)

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls_mean %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls_mean, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls_mean %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args.best_record['val_loss'], train_args.best_record['acc'], train_args.best_record['acc_cls_mean'],
        train_args.best_record['mean_iu'], train_args.best_record['fwavacc'], train_args.best_record['epoch']))
    print('-----------------------------------------------------------------------------------------------------------')

    # [[ add_scalar ]]
    # Adds many scalar data to summary.
    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls_mean', acc_cls_mean, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()