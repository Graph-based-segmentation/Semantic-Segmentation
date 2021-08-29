
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import cv2
import os

import addToConfusionMatrix
from msaspp_dataloader import msasppDataLoader
from cityscapesLabels import labels, id2label
from denseASPP import DenseASPP
from cfgs import DenseASPP121
from tqdm import tqdm

parser = argparse.ArgumentParser(description="msaspp image segmentation inference code")
parser.add_argument('--model_name',             type=str,   help='model name to be trained',                   default='denseaspp-v2')
parser.add_argument('--data_path',              type=str,   help='test data path',                             default=os.getcwd())
parser.add_argument('--input_height',           type=int,   help='input height',                               default=512)
parser.add_argument('--input_width',            type=int,   help='input width',                                default=512)
parser.add_argument('--batch_size',             type=int,   help='train batch size',                           default=8)
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',      default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--num_checkpoint',         type=str,   help='model to be saved after training',           default='model-0029000_mIoU-0.486.pth')
parser.add_argument('--num_seed',               type=int,   help='random seed number',                         default=1)
parser.add_argument('--num_threads',            type=int,   help='number of threads to use for data loading',  default=5)
parser.add_argument('--gpu',                    type=int,   help='GPU id to use',                              default=0)
args = parser.parse_args()

def save_prediction(prediction):
    command = 'mkdir ' + os.path.join(os.getcwd(), "prediction_image")
    os.system(command)
    
    plt.figure(figsize=(40, 10))
    for i in range(19):
        pred = prediction[:, i, :].permute(1, 2, 0)
        pred = pred.detach().cpu().numpy()
        plt.subplot(1, 19, i+1)
        plt.imshow(pred)
        plt.axis('off')
        
    plt.savefig(os.path.join(os.getcwd(), "prediction_image", "pred_imageset.png"), dpi=400, bbox_inches='tight')
    print('Saving done')

def cover_colormap(image):
    COLOR_MAP = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
        (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    
    image = image.data.cpu().numpy()
    # image = image[0, :]
    image = image.argmax(axis=1)[0]
    row, col = image.shape
    dst = np.zeros((row, col, 3), dtype=np.uint8)
    for i in range(19):
        dst[image == i] = COLOR_MAP[i]
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

def generateMatrix(args):
    
    args.evalLabels = []
    for label in labels:
        if (label.id < 0):
            continue
        # we append all found labels, regardless of being ignored
        args.evalLabels.append(label.id)
    maxId = max(args.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)

def getIouScoreForLabel(label, confMatrix, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

def test(args):
    confMatrix      = generateMatrix(args)
    classScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)
    
    dataloader_test = msasppDataLoader(args, mode='test')

    model = DenseASPP(args, model_cfg=DenseASPP121.Model_CFG)
    torch.cuda.set_device(args.gpu)
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    
    model_path = os.path.join(args.checkpoint_path, args.model_name, 'eval_model', args.num_checkpoint)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda(args.gpu)

    with torch.no_grad():
        for data_name, sample in dataloader_test.data:
            image = sample['image'].cuda(args.gpu)
            gt = sample['gt'].cuda(args.gpu)
            
            prediction = model(image)
            
            prediction = F.softmax(prediction, dim=1)
            confMatrix = addToConfusionMatrix.cEvaluatePair(predictionNp, groundTruthNp, confMatrix, args.evalLabels)
            iou_per_class = get_iou(prediction, gt)
            color_prediction = cover_colormap(prediction)
            
            plt.imshow(color_prediction)
            plt.savefig('{}_color.png'.format(*data_name))
            a=1
if __name__ == "__main__":
    test(args)