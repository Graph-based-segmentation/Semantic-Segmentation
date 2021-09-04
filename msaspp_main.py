
import os
import time
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch

# from networks.resnet50_3d_gcn_x5 import RESNET50_3D_GCN_X5
# from torchvision.models.resnet import resnet50
from msaspp_dataloader import msasppDataLoader, colorize_mask
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.backends import cudnn
from denseASPP import DenseASPP
from cfgs import DenseASPP121
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Multi-scale ASPP training')

parser.add_argument('--mode',                   type=str,   help='training and validation mode',    default='train')
parser.add_argument('--model_name',             type=str,   help='model name to be trained',        default='denseaspp-v3')

# Dataset
parser.add_argument('--data_path',              type=str,   help='training data path',              default=os.getcwd())
parser.add_argument('--input_height',           type=int,   help='input height',                    default=512)
parser.add_argument('--input_width',            type=int,   help='input width',                     default=512)

# Training
parser.add_argument('--num_seed',               type=int,   help='random seed number',              default=1)
parser.add_argument('--batch_size',             type=int,   help='train batch size',                default=8)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs',                default=80)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate',           default=3e-4)
parser.add_argument('--weight_decay',           type=float, help='weight decay factor for optimization',                                default=1e-5)
parser.add_argument('--retrain',                type=bool,  help='If used with checkpoint_path, will restart training from step zero',  default=False)
parser.add_argument('--do_eval',                type=bool,  help='Mod to evaluating the training model',                                default=False)

# Preprocessing
parser.add_argument('--random_rotate',          type=bool,  help='if set, will perform random rotation for augmentation',   default=False)
parser.add_argument('--degree',                 type=float, help='random rotation maximum degree',                          default=2.5)

# Log and save
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',               default='')
parser.add_argument('--log_directory',          type=str,   help='directory to save checkpoints and summaries',         default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_freq',               type=int,   help='Logging frequency in global steps',                   default=250)
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


def evaluate_model(val_dataloader, model, criterion):
    eval_iou_sum_score = 0
    eval_loss_sum = 0
    for sampled_eval in tqdm(val_dataloader.data):
        with torch.no_grad():
            eval_image = sampled_eval['image'].cuda(args.gpu, non_blocking=True)
            eval_gt = sampled_eval['gt'].cuda(args.gpu, non_blocking=True)
            
            eval_output = model(eval_image)
            model.eval()
            eval_loss = criterion(eval_output, eval_gt)
            
            eval_output = F.softmax(eval_output, dim=1)
            eval_output = torch.argmax(eval_output, dim=1)
            eval_output = eval_output.contiguous().view(-1)
            eval_gt = eval_gt.contiguous().view(-1)
            
            iou_per_class = []
            for num_class in range(len(val_dataloader.class_names)):
                true_class = (eval_output == num_class)
                true_label = (eval_gt == num_class)
                if true_label.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    union = torch.logical_or(true_class, true_label).sum().float().item()
                    
                    iou = (intersect + 1e-10) / (union + 1e-10)
                    iou_per_class.append(iou)
                    
            eval_iou_sum_score += np.nanmean(iou_per_class)
            eval_loss_sum += eval_loss
            
    return eval_loss_sum, eval_iou_sum_score
    
def main():
    model_filename = args.model_name + '.py'
    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)
    
    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name, 'model')
    os.system(command)
    
    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name, 'eval_model')
    os.system(command)
    
    if args.checkpoint_path == '':
        model_out_path = os.path.join(args.log_directory, args.model_name, model_filename)
        
        command = 'cp denseASPP.py ' + model_out_path
        os.system(command)
        
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        
        command = 'cp msaspp_main.py ' + aux_out_path
        os.system(command)
        
        command = 'cp msaspp_dataloader.py ' + aux_out_path
        os.system(command)
    else:
        args.checkpoint_path = os.path.join(args.checkpoint_path, args.model_name, 'model')
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'
        
        model_out_path = os.path.join(args.log_directory, args.model_name, model_filename)
        command = 'cp ' + os.path.join(loaded_model_dir, loaded_model_filename) + ' ' + model_out_path
        os.system(command)
        
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
    
    model_just_loaded = False
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
        model_just_loaded = True
            
    if args.retrain:
        global_step = 0
        
    cudnn.benchmark = True
    dataloader = msasppDataLoader(args, mode='train')
    val_dataloader = msasppDataLoader(args, mode='val')
    
    # Logging
    writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataloader.ignore_label).cuda()
    start_time = time.time()
    duration = 0
    
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    
    mIoU_list = []
    train_loss_list = []
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
                print('[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}], lr: {:.12f}, train loss: {:.12f}'.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1
                
            train_loss_list.append(loss)
            duration += time.time() - befor_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    # print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | Average train loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, sum(train_loss_list)/len(train_loss_list), time_sofar, training_time_left))
                
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('loss', loss, global_step=global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step=global_step)
                    
                    output = output.data.cpu().numpy()
                    output = output.argmax(axis=1)
                    for i in range(args.batch_size):
                        color_gt = colorize_mask(output[i, :]).convert('RGB')
                        color_gt = transforms.ToTensor()(color_gt)
                        writer.add_image('segmentation_gt/image/{}'.format(i), torch.unsqueeze(sample_gt, 1)[i, :].data, global_step=global_step)
                        writer.add_image('segmentation_est/image/{}'.format(i), color_gt.data, global_step=global_step)
                    writer.flush()
                
                checkpoint = {'global_step': global_step,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'model', 'model-{:07d}.pth'.format(global_step)))
                
            if global_step and global_step % args.save_freq == 0 and not model_just_loaded:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    eval_loss_sum, eval_iou_sum_score = evaluate_model(val_dataloader, model, criterion)
                    mIoU = eval_iou_sum_score / len(val_dataloader.data)
                    mIoU_list.append(mIoU)
                    
                    idx_max = mIoU_list.index(max(mIoU_list))
                    if idx_max == 0 and idx_max == (global_step // args.save_freq - 1):
                        print_string = 'GPU: {} | Epoch: {}/{} | Average train loss: {:.5f} | Average validation loss: {:.5f} | evaluation mIoU: {:.5f}'
                        print(print_string.format(args.gpu, epoch, args.num_epochs, sum(train_loss_list)/len(train_loss_list), eval_loss_sum/len(val_dataloader.data), mIoU_list[idx_max]))
                        checkpoint = {'global_step': global_step,
                                      'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'eval_mIoU': mIoU_list[idx_max]}
                        
                        torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'eval_model', 'model-{:07d}_mIoU-{:.3f}.pth'.format(global_step, mIoU_list[idx_max])))

                    elif idx_max == 0 and idx_max != (global_step // args.save_freq - 1):
                        pass
                    
                    elif idx_max !=0 and idx_max == (global_step // args.save_freq - 1):
                        print_string = 'GPU: {} | Epoch: {}/{} | Average train loss: {:.5f} | Average validation loss: {:.5f} | evaluation mIoU: {:.5f}'
                        print(print_string.format(args.gpu, epoch, args.num_epochs, sum(train_loss_list)/len(train_loss_list), eval_loss_sum/len(val_dataloader.data), max(mIoU_list)))
                        checkpoint = {'global_step': global_step,
                                      'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'eval_mIoU': max(mIoU_list)}
                    
                        torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'eval_model', 'model-{:07d}_mIoU-{:.3f}.pth'.format(global_step, max(mIoU_list))))
                model.train()
            
            model_just_loaded = False
            global_step += 1
        epoch += 1

    checkpoint = {'global_step': global_step,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'model', 'model-{:07d}.pth'.format(global_step)))
    
if __name__ == '__main__':
    main()
