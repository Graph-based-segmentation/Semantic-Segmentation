from PIL import Image
from cfgs import DenseASPP121
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from msaspp_dataloader import msasppDataLoader
from models.DenseASPP_boundary_depthwise import *
from My_train.misc import check_mkdir, AverageMeter
from collections import OrderedDict

import os
import time
import timeit
import shutil
# import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import My_train.joint_transforms as joint_transforms
import My_train.transforms as extended_transforms
import argparse
import torch.nn.functional


parser = argparse.ArgumentParser(description='Multi-scale ASPP training')

# Dataset
parser.add_argument('--input_height',           type=int,   help='input height', default=512)
parser.add_argument('--input_width',            type=int,   help='input width', default=512)
parser.add_argument('--data_path',              type=str,   help='training data path', default='D:\\PycharmProjects\\Personal_project\\Cityscapes_dataset')

# Training
parser.add_argument('--mode',                   type=str,   help='training and test mode',  default='train')
parser.add_argument('--train_batch_size',       type=int,   help='train batch size', default=4)
parser.add_argument('--val_batch_size',         type=int,   help='validation batch size', default=4)
parser.add_argument('--num_threads',            type=int,   help='number of threads to use for data loading', default=12)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate', default=3e-4)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs', default=100)
parser.add_argument('--weight_decay',           type=float, help='weight decay', default=1e-5)
parser.add_argument('--print_frequency',        type=int,   help='print frequency', default=10)
parser.add_argument('--val_save_to_img_file',   type=bool,  help='save validation image file', default=True)
parser.add_argument('--val_img_sample_rate',    type=float, help='randomly sample some validation results to display', default=0.05)

# Preprocessing
parser.add_argument('--random_rotate',          type=bool,  help='if set, will perform random rotation for augmentation', default=True)
parser.add_argument('--degree',                 type=float, help='random rotation maximum degree',  default=2.5)
# Log and save
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',
                    default='/home/mk/Semantic_Segmentation/DenseASPP-master/pretrained_model/densenet121.pth')
parser.add_argument('--GPU',                    type=int,   help='the number of GPU', default=1)
parser.add_argument('--model_freq',             type=int,   help='save the model', default=100)

args = parser.parse_args()

cudnn.benchmark = True

def poly_lr_scheduler(init_lr, epoch, maxEpoch=args.num_epochs, power=0.9):
    "init_lr      : base learning rate \
    iter          : current iteration \
    lr_decay_iter : how frequently decay occurs, default is 1 \
    power         : polynomial power"
    lr = init_lr * ((1 - epoch / maxEpoch) ** power)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    return lr

def main():
    net = DenseASPP_boundary_depthwise(model_cfg=DenseASPP121.Model_CFG).cuda()

    if len(args.checkpoint_path) == 0:
        curr_epoch = 1
        # Initializing 'best_record'
        args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        # load the pretrained model
        print('training resumes from: ', args.checkpoint_path)
        # lambda ==> argument: manipulate(argument)
        pretrained_weight = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        """ map_location = lambda storage, loc: storage--> Load all tensors onto the CPU, using a function"""
        new_state_dict = OrderedDict()
        model_dict = net.state_dict()
        for key, value in pretrained_weight.items():
            name = key
            new_state_dict[name] = value
            if name.find('norm') >= 9:
               # print('norm contained from pretrained_weight : ', name)
               value.requires_grad = False
            # if name.find('conv0') >= 9:
            #     print('norm contained from pretrained_weight : ', name)
            #     value.requires_grad = False

        # new_state_dict.pop('features.conv0.weight')
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

    # ---------------------------------- [[ data - augmentation ]] ---------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # [[joint_transforms]]
    # both raw image and gt are transformed by data-augmentation
    train_joint_transform = joint_transforms.Compose([
        # joint_transforms.ImageScaling(size=[0.5, 2.0]),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomSizedCrop(size=args.input_width),
    ])

    # transform : To preprocess images
    # Compose : if there are a lot of preprocessed images, compose plays a role as collector in a single space.
    input_transform = standard_transforms.Compose([
        # Colorjitter.ColorJitter(brightness=[-10, 10]),
        standard_transforms.ColorJitter(brightness=0.5),
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*my_mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    train_set = msaspp_dataloader.CityScapes('fine', 'train', joint_transform=train_joint_transform,
                                             transform=input_transform, target_transform=target_transform)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_threads, shuffle=True)

    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    weight = [[5.0, 5.0, 5.0, 5.1, 5.0,
               5.0, 5.75, 5.75, 5.55, 5.55,
               5.0, 5.0, 5.0, 5.5, 5.0, 5.2, 5.0,
               5.0, 5.4]]
    class_weight = torch.FloatTensor(weight)
    weighted_criterion = torch.nn.CrossEntropyLoss(ignore_index=msaspp_dataloader.ignore_label,
                                                   weight=class_weight).cuda()
    # unweighted_criterion = torch.nn.CrossEntropyLoss(ignore_index=segmentation_dataloader.ignore_label).cuda()

    num_training_samples = len(train_set)
    steps_per_epoch = np.ceil(num_training_samples / args.train_batch_size).astype(np.int32)
    num_total_steps = args.num_epochs * steps_per_epoch

    print("total number of samples: {}".format(num_training_samples))
    print("total number of steps  : {}".format(num_total_steps))

    # COUNT_PARAMS
    total_num_paramters = 0
    for param in net.parameters():
        total_num_paramters += np.array(list(param.size())).prod()

    print("number of trainable parameters: {}".format(total_num_paramters))

    for epoch in range(curr_epoch, args.num_epochs + 1):
        lr_ = poly_lr_scheduler(init_lr=args.learning_rate, epoch=epoch - 1)
        optimizer = optim.Adam(net.parameters(), lr=lr_, weight_decay=args.weight_decay)

        # train(train_loader, net, criterion, optimizer, epoch, args, huber_loss)
        train(train_loader, net, weighted_criterion, optimizer, epoch, args)

    torch.save(net.state_dict(), os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet,
                                              'model-{}'.format(total_num_paramters) + '.pkl'))
    print('Training Done!!')

def train(train_loader, net, weighted_criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    # loss_seg = AverageMeter()
    # loss_seg_v2 = AverageMeter()

    # curr_iter : total dataset per epoch
    curr_iter = (epoch - 1) * len(train_loader)
    index = 0

    start_time = time.time()
    # bce_loss = BCEWithLogitsLoss2d()
    # loss_D_value = 0

    net.train()
    for step, data in enumerate(train_loader):
        predictions_all = []
        visual = []

        inputs, labels, gt_pyramid = data
        assert inputs.size()[2:] == labels.size()[1:]
        # ignore_mask = (labels.numpy() == 255)
        N = inputs.size(0)

        # D_gt_v = Variable(one_hot(labels)).cuda()
        # ignore_mask_gt = (labels.numpy() == 255)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == msaspp_dataloader.num_classes
        before_op_time = timeit.default_timer()

        # output image pyramid
        output_np = outputs.data.cpu().numpy()
        output_pil = Image.fromarray(output_np.astype(np.uint8)).convert('P')
        # weighted_criterion()
        loss = weighted_criterion(outputs, labels)

        # LOSS = loss + 0.05 * semi_loss + 0.01 * semi_loss_v2
        duration = timeit.default_timer() - before_op_time

        # semi_loss.backward()
        # semi_loss_v2.backward()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start_time

        train_loss.update(loss.data[0], N)
        # loss_seg.update(semi_loss.data[0], N)
        # loss_seg_v2.update(semi_loss_v2.data[0], N)
        curr_iter += 1

        writer.add_scalar('train_loss',train_loss.avg, curr_iter)
        # writer.add_scalar('loss_seg', loss_seg.avg, curr_iter)
        # writer.add_scalar('loss_seg_v2', loss_seg_v2.avg, curr_iter)
        # writer.add_scalar('LOSS', LOSS, curr_iter)

        if (step + 1) % train_args.print_frequency == 0:
            examples_time = args.train_batch_size / duration
            print('epoch: %d | iter: %d / %d | train loss: %.5f | |examples/s: %4.2f | time_elapsed: %.5f''s' %
                  (epoch, step + 1, len(train_loader), train_loss.avg, examples_time, batch_time))

            # print('epoch: %d | iter: %d / %d | loss_D = 4:%.3f | train loss: %.5f |examples/s: %4.2f | time_elapsed: %.5f''s' %
            #       (epoch, step + 1, len(train_loader), loss_D_value, train_loss.avg,examples_time, batch_time))

            # SAVE THE IMAGES AND THE MODEL
            if (step + 1) % train_args.model_freq == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet,
                                                          'model-{}'.format(step + 1) + '.pkl'))
                data_transform = standard_transforms.ToTensor()

                np_outputs = outputs.data.cpu().numpy()
                result = np_outputs.argmax(axis=1)
                predictions_all.append(result)
            else:
                continue

            predictions_all = np.concatenate(predictions_all)
            for idx, data in enumerate(predictions_all):
                predictions_pil = msaspp_dataloader.colorize_mask(data)
                predictions = data_transform(predictions_pil.convert('RGB'))
                visual.extend([predictions])

            visual = torch.stack(visual, 0)
            visual = vutils.make_grid(visual, nrow=2, padding=0)
            # result = np_outputs.argmax(axis=1)[0]
            # row, col = result.shape
            # dst = np.zeros((row, col, 3), dtype=np.uint8)
            #
            # for i in range(19):
            #     dst[result == i] = COLOR_MAP[i]
            # dst = np.array(dst, dtype=np.uint8)
            # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            # if not os.path.exists(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'prediction')):
            #     os.makedirs(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'prediction'))
            #
            # cv2.imwrite(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'prediction/%06d.png' %
            #                          epoch), dst)
            writer.add_image('OUTPUT_IMAGE{}'.format(epoch), visual, global_step=step+1)

    with open(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'LR_v0{}_{}.txt'.format(x,version)), 'a') as LRtxt:
        LRtxt.write("index : {}, epoch : {}, learning rate : {: f}".format(index, epoch, optimizer.param_groups[0]['lr']) + '\n')
        index += 1

if __name__ == '__main__':

    dataloader = msasppDataLoader(args)
    for step, sample_batched in enumerate(dataloader.data):
        print(step, sample_batched)
    # x = 1
    # version = '09'
    #
    # ckpt_path = '../../ckpt'
    # ImageNet = 'ImageNet/DenseNet121_v3'
    # exp_name_ImageNet = 'segImageNet_v0{}_{}'.format(x, version)
    #
    # writer = SummaryWriter(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet))
    #
    # check_mkdir(ckpt_path)
    # check_mkdir(os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet))
    # open(os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet, str(datetime.datetime.now()) + '.txt'),
    #      'w').write(
    #     str(args) + '\n\n')
    #
    # src = "/home/mk/Semantic_Segmentation/DenseASPP-master/My_train/msaspp_main.py"
    # src_model = "/home/mk/Semantic_Segmentation/DenseASPP-master/models/DenseASPP_boundary_depthwise.py"

    # copy_path = os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet,
    #                          "segmentation_main_v3_" + "v0{}_{}.py".format(x, version))
    # model_copy_path = os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet,
    #                                "DenseASPP_boundary_depthwise" + "v0{}_{}.py".format(x, version))
    #
    # shutil.copy(src, copy_path)
    # shutil.copy(src_model, model_copy_path)

    GPU_ID = args.GPU

    main()
