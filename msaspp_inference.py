
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import cv2
import os

from msaspp_dataloader import msasppDataLoader
from cityscapesLabels import labels, id2label
from denseASPP import DenseASPP
from cfgs import DenseASPP121
from tqdm import tqdm

parser = argparse.ArgumentParser(description="msaspp image segmentation inference code")
parser.add_argument('--model_name',             type=str,   help='model name to be trained',                         default='denseaspp-v3')
parser.add_argument('--data_path',              type=str,   help='test data path',                                   default=os.getcwd())
parser.add_argument('--input_height',           type=int,   help='input height',                                     default=512)
parser.add_argument('--input_width',            type=int,   help='input width',                                      default=512)
parser.add_argument('--batch_size',             type=int,   help='train batch size',                                 default=8)
parser.add_argument('--log_directory',          type=str,   help='directory to save checkpoints and summaries',      default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--num_checkpoint',         type=str,   help='model to be saved after training',                 default='model-0029500_mIoU-0.445.pth')
parser.add_argument('--num_seed',               type=int,   help='random seed number',                               default=1)
parser.add_argument('--num_threads',            type=int,   help='number of threads to use for data loading',        default=5)
parser.add_argument('--gpu',                    type=int,   help='GPU id to use',                                    default=0)
args = parser.parse_args()


def check_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def fill_colormap(prediction):
    COLOR_MAP = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                 (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    
    prediction = prediction.data.cpu().numpy()
    # prediction = prediction[0, :]
    prediction = prediction.argmax(axis=1)[0]
    row, col = prediction.shape
    dst = np.zeros((row, col, 3), dtype=np.uint8)
    for i in range(19):
        dst[prediction == i] = COLOR_MAP[i]
    dst = np.array(dst, dtype=np.uint8)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    
    return dst

def get_iou(prediction, gt):
    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.contiguous().view(-1)
    gt = gt.contiguous().view(-1)
    iou_per_class = []
    for num_class in range(19):
        true_class = (prediction == num_class)
        true_label = (gt == num_class)
        if true_label.long().sum().item() == 0:
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()
            
            iou = (intersect + 1e-10) / (union + 1e-10)
            iou_per_class.append(iou)
    
    return iou_per_class

def test(args):
    dataloader_eval = msasppDataLoader(args, mode='eval')

    model = DenseASPP(args, model_cfg=DenseASPP121.Model_CFG)
    torch.cuda.set_device(args.gpu)
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    
    model_path = os.path.join(args.log_directory, args. model_name, 'eval_model', args.num_checkpoint)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda(args.gpu)

    with torch.no_grad():
        for idx, (name, sample) in enumerate(dataloader_eval.data):
            image = sample['image'].cuda(args.gpu)
            gt = sample['gt'].cuda(args.gpu)
            
            prediction = model(image)
                
            prediction = F.log_softmax(prediction, dim=1)
            iou_per_class = get_iou(prediction, gt)
            if prediction.size()[0] < 1024:
                prediction = F.upsample(prediction, size=(1024, 2048), mode='bilinear')
            mIoU = np.nanmean(iou_per_class)
            
            color_prediction = fill_colormap(prediction)
            print("idx: {}, data name: {}, mIoU: {}".format(idx+1, *name, mIoU))
            
            check_directory(path=os.path.join(args.log_directory, args.model_name, 'prediction_image'))
            plt.figure(figsize=(18, 5))
            plt.suptitle("name: {} - mIoU: {}".format(*name, mIoU))
            plt.subplot(1, 2, 1)
            plt.imshow(color_prediction)
            plt.axis('off')
            
            plt_img = image.squeeze().permute(1, 2, 0)
            plt_img = plt_img.data.cpu().numpy()
            plt.subplot(1, 2, 2)
            plt.imshow(plt_img)
            plt.axis('off')
                        
            plt.savefig(os.path.join(args.log_directory, args.model_name, 'prediction_image', '{}_color'.format(*name)), dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    test(args)