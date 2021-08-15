
import os
import time
import timeit
import argparse
import numpy as np
import torch.nn.functional
import torch.multiprocessing as mp
import torch

# from networks.resnet50_3d_gcn_x5 import RESNET50_3D_GCN_X5
# from torchvision.models.resnet import resnet50
from torch.utils.tensorboard import SummaryWriter
from denseASPP import DenseASPP
from cfgs import DenseASPP121
from torch.backends import cudnn
from msaspp_dataloader import msasppDataLoader

parser = argparse.ArgumentParser(description='Multi-scale ASPP training')

parser.add_argument('--mode',                   type=str,   help='training and test mode',          default='train')
parser.add_argument('--model_name',             type=str,   help='model name to be trained',        default='denseaspp')
# parser.add_argument('--model_select',           type=str,   help='Select the model type',   default='resnet')

# Dataset
parser.add_argument('--data_path',              type=str,   help='training data path',              default=os.getcwd())
parser.add_argument('--input_height',           type=int,   help='input height',                    default=512)
parser.add_argument('--input_width',            type=int,   help='input width',                     default=512)

# Training
parser.add_argument('--num_seed',               type=int,   help='random seed number',              default=1)
parser.add_argument('--train_batch_size',       type=int,   help='train batch size',                default=8)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs',                default=80)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate',           default=3e-4)
parser.add_argument('--weight_decay',           type=float, help='weight decay factor for optimization',                                default=1e-5)
# parser.add_argument('--val_batch_size',         type=int,   help='validation batch size', default=4)
# parser.add_argument('--val_save_to_img_file',   type=bool,  help='save validation image file', default=True)
# parser.add_argument('--val_img_sample_rate',    type=float, help='randomly sample some validation results to display', default=0.05)
parser.add_argument('--retrain',                type=bool,  help='If used with checkpoint_path, will restart training from step zero',  default=False)

# Preprocessing
parser.add_argument('--random_rotate',          type=bool,  help='if set, will perform random rotation for augmentation',   default=False)
parser.add_argument('--degree',                 type=float, help='random rotation maximum degree',                          default=2.5)

# Log and save
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',               default='')
parser.add_argument('--log_directory',          type=str,   help='directory to save checkpoints and summaries',         default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_freq',               type=int,   help='Logging frequency in global steps',                   default=100)
parser.add_argument('--save_freq',              type=int,   help='Checkpoint saving frequency in global steps',         default=500)

# Multi-gpu training
parser.add_argument('--gpu',            type=int,  help='GPU id to use', default=0)
parser.add_argument('--rank',           type=int,  help='node rank(tensor dimension)for distributed training', default=0)
parser.add_argument('--dist_url',       type=str,  help='url used to set up distributed training', default='file:///c:/MultiGPU.txt')
parser.add_argument('--dist_backend',   type=str,  help='distributed backend', default='gloo')
parser.add_argument('--num_threads',    type=int,  help='number of threads to use for data loading', default=5)
parser.add_argument('--world_size',     type=int,  help='number of nodes for distributed training', default=1)
parser.add_argument('--multiprocessing_distributed',       help='Use multi-processing distributed training to launch '
                                                                'N process per node, which has N GPUs. '
                                                                'This is the fastest way to use PyTorch for either single node or '
                                                                'multi node data parallel training', default=False)
args = parser.parse_args()

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def main():
    check_folder(args.log_directory)
    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(ngpus_per_node, args)
        
def main_worker(ngpus_per_node, args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # if args.model_select == 'resnet':
    #     model = resnet50(pretrained=True)
    #     del [model.fc, model.avgpool]
        
    # elif args.model_select == 'glore':
    #     model = RESNET50_3D_GCN_X5(num_classes=19, pretrained=False)
    model = DenseASPP(args, model_cfg=DenseASPP121.Model_CFG)
    model.train()
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))
    
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))
    
    if args.distributed:
        a=1
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")
        
    global_step = 0
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu in None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda: {}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
            
    if args.retrain:
        global_step = 0
        
    cudnn.benchmark = True
    dataloader = msasppDataLoader(args)
    
    # Logging
    writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataloader.ignore_label).cuda()
    start_time = time.time()
    duration = 0
    
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    
    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)
            
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            befor_op_time = time.time()
            
            sample_image = sample_batched['image'].cuda(args.gpu, non_blocking=True)
            sample_gt = sample_batched['gt'].cuda(args.gpu, non_blocking=True)
            
            output = model(sample_image)
            
            loss = criterion(output, sample_gt)
            loss.backward()
            
            for param_group in optimizer.param_groups:
                current_lr = args.learning_rate * ((1 - epoch / args.num_epochs) ** 0.9)
                param_group['lr'] = current_lr
            
            optimizer.step()
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1
                
            duration += time.time() - befor_op_time
            if global_step and global_step * args.log_freq == 0:
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, time_sofar, training_time_left))
                
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('loss', loss, global_step=global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step=global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step=global_step)
                    
                    for i in range(args.num_epochs):
                        writer.add_image('segmentation_gt/image/{}'.format(i), sample_gt[i, :, :, :].data, global_step=global_step)
                        writer.add_image('segmentation_est/image/{}'.format(i), output[i, :, :, :].data, global_step=global_step)
                    writer.flush()
                    
            if global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed and (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'model', 'model-{}.pth'.format(global_step)))
                    
#     if len(args.checkpoint_path) == 0:
#         curr_epoch = 1
#         # Initializing 'best_record'
#         args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
#     else:
#         # load the pretrained model
#         print('training resumes from: ', args.checkpoint_path)
#         # lambda ==> argument: manipulate(argument)
#         pretrained_weight = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
#         """ map_location = lambda storage, loc: storage--> Load all tensors onto the CPU, using a function"""
#         new_state_dict = OrderedDict()
#         model_dict = net.state_dict()
#         for key, value in pretrained_weight.items():
#             name = key
#             new_state_dict[name] = value
#             if name.find('norm') >= 9:
#                # print('norm contained from pretrained_weight : ', name)
#                value.requires_grad = False
#             # if name.find('conv0') >= 9:
#             #     print('norm contained from pretrained_weight : ', name)
#             #     value.requires_grad = False

#         # new_state_dict.pop('features.conv0.weight')
#         new_state_dict.pop('features.norm5.weight')
#         new_state_dict.pop('features.norm5.bias')
#         new_state_dict.pop('features.norm5.running_mean')
#         new_state_dict.pop('features.norm5.running_var')
#         new_state_dict.pop('classifier.weight')
#         new_state_dict.pop('classifier.bias')
#         model_dict.update(new_state_dict)
#         net.load_state_dict(model_dict)
#         curr_epoch = 1
#         args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

#     train_joint_transform = joint_transforms.Compose([
#         # joint_transforms.ImageScaling(size=[0.5, 2.0]),
#         joint_transforms.RandomHorizontallyFlip(),
#         joint_transforms.RandomSizedCrop(size=args.input_width),
#     ])

#     input_transform = standard_transforms.Compose([
#         # Colorjitter.ColorJitter(brightness=[-10, 10]),
#         standard_transforms.ColorJitter(brightness=0.5),
#         standard_transforms.ToTensor(),
#         # standard_transforms.Normalize(*my_mean_std)
#     ])

#     target_transform = extended_transforms.MaskToTensor()

#     train_set = msaspp_dataloader.CityScapes('fine', 'train', joint_transform=train_joint_transform,
#                                              transform=input_transform, target_transform=target_transform)

#     train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_threads, shuffle=True)

#     # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

#     weight = [[5.0, 5.0, 5.0, 5.1, 5.0,
#                5.0, 5.75, 5.75, 5.55, 5.55,
#                5.0, 5.0, 5.0, 5.5, 5.0, 5.2, 5.0,
#                5.0, 5.4]]
#     class_weight = torch.FloatTensor(weight)
#     weighted_criterion = torch.nn.CrossEntropyLoss(ignore_index=msaspp_dataloader.ignore_label,
#                                                    weight=class_weight).cuda()
#     # unweighted_criterion = torch.nn.CrossEntropyLoss(ignore_index=segmentation_dataloader.ignore_label).cuda()

#     num_training_samples = len(train_set)
#     steps_per_epoch = np.ceil(num_training_samples / args.train_batch_size).astype(np.int32)
#     num_total_steps = args.num_epochs * steps_per_epoch

#     print("total number of samples: {}".format(num_training_samples))
#     print("total number of steps  : {}".format(num_total_steps))

#     # COUNT_PARAMS
#     total_num_paramters = 0
#     for param in net.parameters():
#         total_num_paramters += np.array(list(param.size())).prod()

#     print("number of trainable parameters: {}".format(total_num_paramters))

#     for epoch in range(curr_epoch, args.num_epochs + 1):
#         lr_ = poly_lr_scheduler(init_lr=args.learning_rate, epoch=epoch - 1)
#         optimizer = optim.Adam(net.parameters(), lr=lr_, weight_decay=args.weight_decay)

#         # train(train_loader, net, criterion, optimizer, epoch, args, huber_loss)
#         train(train_loader, net, weighted_criterion, optimizer, epoch, args)

#     torch.save(net.state_dict(), os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet,
#                                               'model-{}'.format(total_num_paramters) + '.pkl'))
#     print('Training Done!!')

# def train(train_loader, net, weighted_criterion, optimizer, epoch, train_args):
#     train_loss = AverageMeter()
#     # loss_seg = AverageMeter()
#     # loss_seg_v2 = AverageMeter()

#     # curr_iter : total dataset per epoch
#     curr_iter = (epoch - 1) * len(train_loader)
#     index = 0

#     start_time = time.time()
#     # bce_loss = BCEWithLogitsLoss2d()
#     # loss_D_value = 0

#     net.train()
#     for step, data in enumerate(train_loader):
#         predictions_all = []
#         visual = []

#         inputs, labels, gt_pyramid = data
#         assert inputs.size()[2:] == labels.size()[1:]
#         # ignore_mask = (labels.numpy() == 255)
#         N = inputs.size(0)

#         # D_gt_v = Variable(one_hot(labels)).cuda()
#         # ignore_mask_gt = (labels.numpy() == 255)
#         inputs = Variable(inputs).cuda()
#         labels = Variable(labels).cuda()

#         optimizer.zero_grad()

#         outputs = net(inputs)
#         assert outputs.size()[2:] == labels.size()[1:]
#         assert outputs.size()[1] == msaspp_dataloader.num_classes
#         before_op_time = timeit.default_timer()

#         # output image pyramid
#         output_np = outputs.data.cpu().numpy()
#         output_pil = Image.fromarray(output_np.astype(np.uint8)).convert('P')
#         # weighted_criterion()
#         loss = weighted_criterion(outputs, labels)

#         # LOSS = loss + 0.05 * semi_loss + 0.01 * semi_loss_v2
#         duration = timeit.default_timer() - before_op_time

#         # semi_loss.backward()
#         # semi_loss_v2.backward()
#         loss.backward()
#         optimizer.step()
#         batch_time = time.time() - start_time

#         train_loss.update(loss.data[0], N)
#         # loss_seg.update(semi_loss.data[0], N)
#         # loss_seg_v2.update(semi_loss_v2.data[0], N)
#         curr_iter += 1

#         writer.add_scalar('train_loss',train_loss.avg, curr_iter)
#         # writer.add_scalar('loss_seg', loss_seg.avg, curr_iter)
#         # writer.add_scalar('loss_seg_v2', loss_seg_v2.avg, curr_iter)
#         # writer.add_scalar('LOSS', LOSS, curr_iter)

#         if (step + 1) % train_args.print_frequency == 0:
#             examples_time = args.train_batch_size / duration
#             print('epoch: %d | iter: %d / %d | train loss: %.5f | |examples/s: %4.2f | time_elapsed: %.5f''s' %
#                   (epoch, step + 1, len(train_loader), train_loss.avg, examples_time, batch_time))

#             # print('epoch: %d | iter: %d / %d | loss_D = 4:%.3f | train loss: %.5f |examples/s: %4.2f | time_elapsed: %.5f''s' %
#             #       (epoch, step + 1, len(train_loader), loss_D_value, train_loss.avg,examples_time, batch_time))

#             # SAVE THE IMAGES AND THE MODEL
#             if (step + 1) % train_args.model_freq == 0:
#                 torch.save(net.state_dict(), os.path.join(ckpt_path, 'Model', ImageNet, exp_name_ImageNet,
#                                                           'model-{}'.format(step + 1) + '.pkl'))
#                 data_transform = standard_transforms.ToTensor()

#                 np_outputs = outputs.data.cpu().numpy()
#                 result = np_outputs.argmax(axis=1)
#                 predictions_all.append(result)
#             else:
#                 continue

#             predictions_all = np.concatenate(predictions_all)
#             for idx, data in enumerate(predictions_all):
#                 predictions_pil = msaspp_dataloader.colorize_mask(data)
#                 predictions = data_transform(predictions_pil.convert('RGB'))
#                 visual.extend([predictions])

#             visual = torch.stack(visual, 0)
#             visual = vutils.make_grid(visual, nrow=2, padding=0)
#             # result = np_outputs.argmax(axis=1)[0]
#             # row, col = result.shape
#             # dst = np.zeros((row, col, 3), dtype=np.uint8)
#             #
#             # for i in range(19):
#             #     dst[result == i] = COLOR_MAP[i]
#             # dst = np.array(dst, dtype=np.uint8)
#             # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
#             # if not os.path.exists(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'prediction')):
#             #     os.makedirs(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'prediction'))
#             #
#             # cv2.imwrite(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'prediction/%06d.png' %
#             #                          epoch), dst)
#             writer.add_image('OUTPUT_IMAGE{}'.format(epoch), visual, global_step=step+1)

#     with open(os.path.join(ckpt_path, 'TensorboardX', ImageNet, exp_name_ImageNet, 'LR_v0{}_{}.txt'.format(x,version)), 'a') as LRtxt:
#         LRtxt.write("index : {}, epoch : {}, learning rate : {: f}".format(index, epoch, optimizer.param_groups[0]['lr']) + '\n')
#         index += 1

if __name__ == '__main__':
    main()
