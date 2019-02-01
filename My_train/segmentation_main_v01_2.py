import datetime
import os
import random
import time
import timeit
import shutil
import torchvision.models as models
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import My_train.joint_transforms as joint_transforms
import My_train.transforms as extended_transforms
import My_train.color_transforms as Colorjitter
# import My_train.size_transforms as Mytransforms
from My_train import size_transforms as Mytransforms
import argparse
import torch.nn.functional

from PIL import Image
from cfgs import DenseASPP121
from cfgs import DenseASPP161
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from My_train import segmentation_dataloader

# from models.DenseASPP_v2 import *
from models.DenseASPP_v3 import *
# from models.DenseASPP import *

from My_train.misc import check_mkdir, evaluate, AverageMeter, get_iou
from collections import OrderedDict

parser = argparse.ArgumentParser(description='DenseASPP training')
parser.add_argument('--input_height',           type=int,   help='input height', default=512)
parser.add_argument('--input_width',            type=int,   help='input width', default=512)
parser.add_argument('--train_batch_size',       type=int,   help='train batch size', default=4)
parser.add_argument('--val_batch_size',         type=int,   help='validation batch size', default=4)
parser.add_argument('--num_threads',            type=int,   help='number of threads to use for data loading', default=12)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate', default=3e-4)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs', default=80)
parser.add_argument('--weight_decay',           type=float, help='weight decay', default=1e-5)
parser.add_argument('--print_frequency',        type=int,   help='print frequency', default=10)
parser.add_argument('--val_save_to_img_file',   type=bool,  help='save validation image file', default=True)
parser.add_argument('--val_img_sample_rate',    type=float, help='randomly sample some validation results to display', default=0.05)
parser.add_argument('--checkpoint_path',        type=str,   help='path ro a specific checkpoint to load',
                    default='/home/mk/Semantic_Segmentation/DenseASPP-master/pretrained_model/densenet121.pth')
parser.add_argument('--GPU',                    type=int,   help='the number of GPU', default=1)

args = parser.parse_args()

def poly_lr_scheduler(optimizer, init_lr, epoch, maxEpoch=args.num_epochs, power=0.9):


    "init_lr      : base learning rate \
    iter          : current iteration \
    lr_decay_iter : how frequently decay occurs, default is 1 \
    power         : polynomial power"
    lr = init_lr * ((1 - epoch / maxEpoch) ** power)
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
        # lambda ==> argument: manipulate(argument)
        pretrained_weight = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        """ map_location = lambda storage, loc: storage--> Load all tensors onto the CPU, using a function"""
        new_state_dict = OrderedDict()
        model_dict = net.state_dict()
        for key, value in pretrained_weight.items():
            name = key
            new_state_dict[name] = value
            if name.find('norm') >= 9:
               print('norm contained from pretrained_weight : ', name)
               value.requires_grad = False
            # if name.find('conv0') >= 9:
            #     print('norm contained from pretrained_weight : ', name)
            #     value.requires_grad = False

        new_state_dict.pop('features.norm5.weight')
        new_state_dict.pop('features.norm5.bias')
        new_state_dict.pop('features.norm5.running_mean')
        new_state_dict.pop('features.norm5.running_var')
        new_state_dict.pop('classifier.weight')
        new_state_dict.pop('classifier.bias')
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)
        # pretrained_dict = {key: value for key, value in pretrained_dict.items() if key in model_dict}
        # model_dict.update(pretrained_dict)
        # pretrained_dict = {key: value for key, value in pretrained_dict.items() if key != 'classifier.weight' or 'classifier.bias'}

        # model.load_state_dict(model_dict, strict=False)
        # model.load_state_dict(new_pretrained_dict, strict=False)
        curr_epoch = 1
        args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

    cudnn.benchmark = True
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    val_mean_std = ([0.290101, 0.328081, 0.286964], [0.182954, 0.186566, 0.184475])

    # ---------------------------------- [[ data - augmentation ]] ---------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # [[joint_transforms]]
    # both raw image and gt are transformed by data-augmentation
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomSizedCrop(size=args.input_width),
    ])

    val_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomSizedCrop(size=args.input_width),
    ])

    # transform : To preprocess images
    # Compose : if there are a lot of preprocessed images, compose plays a role as collector in a single space.
    input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(hue=0.1),
        # Colorjitter.ColorJitter(brightness=[-0.5, 0.5]),
        # Colorjitter.AdjustBrightness(brightness=10),
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*val_mean_std)
    ])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(hue=0.1),
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*val_mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        # """[[ ToPILImage() ]]"""
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

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=segmentation_dataloader.ignore_label).cuda()

    for epoch in range(curr_epoch, args.num_epochs + 1):
        # net.train()
        train(train_loader, net, criterion, optimizer, epoch, args, train_set)
        net.eval()
        validate(val_loader, net, criterion, optimizer, epoch, args, restore_transform, visualize)

    print('Training Done!!')

def train(train_loader, net, criterion, optimizer, epoch, train_args, train_set):
    # batch_time = AverageMeter()
    train_loss = AverageMeter()
    # examples_time = AverageMeter()

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
    index = 0

    start_time = time.time()
    net.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        """zero_grad() : Sets gradients of all model parameters to zero."""
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == segmentation_dataloader.num_classes

        before_op_time = timeit.default_timer()
        # loss = torch.nn.functional.cross_entropy(input=outputs, target=labels, ignore_index=segmentation_dataloader.ignore_label)
        loss = criterion(outputs, labels)
        duration = timeit.default_timer() - before_op_time

        loss.backward()
        optimizer.step()
        batch_time = time.time() - start_time

        train_loss.update(loss.data[0], N)
        curr_iter += 1

        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args.print_frequency == 0:
            examples_time = args.train_batch_size / duration
            print('epoch: %d | iter: %d / %d | train loss: %.5f | examples/s: %4.2f | time_elapsed: %.5f''s' %
                  (epoch, i + 1, len(train_loader), train_loss.avg, examples_time, batch_time))

        poly_lr_scheduler(optimizer=optimizer, init_lr=args.learning_rate, epoch=epoch-1)
        with open(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'LR_v0{}_{}.txt'.format(x,version)), 'a') as LRtxt:
            LRtxt.write("index : {}, epoch : {}, learning rate : {: f}".format(index, epoch, optimizer.param_groups[0]['lr']) + '\n')
            index += 1

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    val_loss = AverageMeter()

    inputs_all, gts_all, predictions_all = [], [], []
    # net.eval()
    total_miou = 0.0
    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()
        outputs = net(inputs)
        # predictions = outputs.data.max(1)[1].squeeze(1).cpu().numpy()
        predictions = outputs.data.cpu().numpy()
        predictions = predictions.argmax(axis=1)
        # predictions = torch.max(outputs, 1)[1]
        val_loss.update(criterion(outputs, gts).data[0] / N, N)
        # validation_loss = torch.nn.functional.cross_entropy(input=outputs, target=gts, ignore_index=segmentation_dataloader.ignore_label)
        # val_loss.update(validation_loss.data[0] / N, N)
        for i in inputs:
            if random.random() > train_args.val_img_sample_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(i.data.cpu())
        # total_miou += get_iou(pred=predictions, gt=gts, n_classes=segmentation_dataloader.num_classes)
        gts_all.append(gts.data.cpu().numpy())
        predictions_all.append(predictions)

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)
    # print('mean_IoU by confusion_matrix : ',
    #       mean_iou(predictions=predictions_all, gts=gts_all, num_class=segmentation_dataloader.num_classes))
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

    # print(ValEvaluate(predictions_all, gts_all, segmentation_dataloader.num_classes))
    # print(mean_IoU(predictions_all, gts_all))
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
        snapshot_name = 'epoch_%d_loss_%.6f_acc_%.6f_acc-cls_%.6f_mean-iu_%.6f_fwavacc_%.6f_lr_%.10f' % (
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
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls_mean %.5f], [mean_iu %.6f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls_mean, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls_mean %.5f], [mean_iu %.6f], [fwavacc %.5f], [epoch %d]' % (
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

    return val_loss.avg


if __name__ == '__main__':
    x = 2
    version = '0_2'

    ckpt_path = '../../ckpt'
    ImageNet = 'ImageNet/DenseNet121_v2'
    exp_name_ImageNet = 'segImageNet_v0{}_{}'.format(x, version)

    # [[ SummaryWriter]]
    # Writes 'Summary' directly to event files.
    # writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))
    writer = SummaryWriter(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet))

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet))
    open(os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet, str(datetime.datetime.now()) + '.txt'),
         'w').write(
        str(args) + '\n\n')

    src = "/home/mk/Semantic_Segmentation/DenseASPP-master/My_train/segmentation_main2.py"
    src_model = "/home/mk/Semantic_Segmentation/DenseASPP-master/models/DenseASPP_v3.py"

    copy_path = os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet,
                             "segmentation_main2_" + "v_0{}_{}.py".format(x, version))
    model_copy_path = os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet,
                                   "DenseASPP" + "v_0{}_{}.py".format(x, version))

    shutil.copy(src, copy_path)
    shutil.copy(src_model, model_copy_path)

    GPU_ID = args.GPU

    main()