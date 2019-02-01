import os
import cv2
import torch
import numpy
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict


IS_MULTISCALE = True
N_CLASS = 19

COLOR_MAP = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

# COLOR_MAP = [7, 8, 11, 12, 13, 17,
#              19, 20, 21, 22, 23, 24,
#              25, 26, 27, 28, 31, 32, 33]


inf_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.8]
# inf_scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.290101, 0.328081, 0.286964],
                                                           [0.182954, 0.186566, 0.184475])])
# data_transforms = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406],
#                                                            [0.229, 0.224, 0.225])])
# data_transforms = transforms.Compose([transforms.ToTensor()])

class Inference(object):

    def __init__(self, model_name, model_path):
        self.seg_model = self.__init_model(model_name, model_path, is_local=False)

    def __init_model(self, model_name, model_path, is_local=False):
        if model_name == 'DenseASPP121':
            from cfgs.DenseASPP121 import Model_CFG
            from models.DenseASPP_boundary import DenseASPP_boundary
        else:
            from cfgs.DenseASPP161 import Model_CFG
            from models.DenseASPP_boundary import DenseASPP_boundary

        seg_model = DenseASPP_boundary(Model_CFG, n_class=N_CLASS, output_stride=8)
        self.__load_weight(seg_model, model_path, is_local=is_local)
        seg_model.eval()
        seg_model = seg_model.cuda()

        return seg_model

    def folder_inference(self, img_dir, bound_dir, is_multiscale=True):
        base_path = '/home/mk/Semantic_Segmentation/DenseASPP-master/Results'
        save_path = "single_inference"

        folders = sorted(os.listdir(img_dir))

        for f in folders:
            if not os.path.exists(os.path.join(base_path, 'Densenet121_prediction_image_eval', save_path, f)):
                os.makedirs(os.path.join(base_path, 'Densenet121_prediction_image_eval', save_path, f))

            read_path = os.path.join(img_dir, f)
            bound_read_path = os.path.join(bound_dir, f)
            # if f == 'Hanyang_20190108_2':
            #     read_path = os.path.join(img_dir, f)
            # else:
            #     continue
            names = sorted(os.listdir(read_path))
            for idx, n in enumerate(names):
                if not n.endswith(".png"):
                    continue
                print(n)
                read_name = os.path.join(read_path, n)
                bound_read_name = os.path.join(bound_read_path, n)

                img = Image.open(read_name)
                bound_img = Image.open(bound_read_name)

                if is_multiscale:
                    pre = self.multiscale_inference(img)
                else:
                    pre = self.single_inference(img, bound_img)

                # mask = self.__pre_to_train_id(pre)
                mask = self.__pre_to_img(pre)
                # # cv2.imshow('DenseASPP', mask)
                cv2.imwrite(os.path.join(base_path, 'Densenet121_prediction_image_eval', save_path, f, '{}_Mask.png'.format(n[:-4])), mask)
                cv2.waitKey(1)

    def multiscale_inference(self, test_img):
        h, w = test_img.size
        pre = []
        for scale in inf_scales:
            img_scaled = test_img.resize((int(h * scale), int(w * scale)), Image.CUBIC)
            pre_scaled = self.single_inference(img_scaled, is_flip=False)
            pre.append(pre_scaled)

            img_scaled = img_scaled.transpose(Image.FLIP_LEFT_RIGHT)
            pre_scaled = self.single_inference(img_scaled, is_flip=True)
            pre.append(pre_scaled)

        pre_final = self.__fushion_avg(pre)

        return pre_final

    def single_inference(self, test_img, test_bound_img, is_flip=False):
        image = Variable(data_transforms(test_img).unsqueeze(0).cuda(), volatile=True)
        pre = self.seg_model.forward(image)

        if pre.size()[0] < 1024:
            pre = F.upsample(pre, size=(1024, 2048), mode='bilinear')

        pre = F.log_softmax(pre, dim=1)
        pre = pre.data.cpu().numpy()

        if is_flip:
            tem = pre[0]
            tem = tem.transpose(1, 2, 0)
            tem = numpy.fliplr(tem)
            tem = tem.transpose(2, 0, 1)
            pre[0] = tem

        return pre

    @staticmethod
    def __fushion_avg(pre):
        pre_final = 0
        for pre_scaled in pre:
            pre_final = pre_final + pre_scaled
        pre_final = pre_final / len(pre)
        return pre_final

    @staticmethod
    def __load_weight(seg_model, model_path, is_local=True):
        print("loading pre-trained weight")
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)

        if is_local:
            seg_model.load_state_dict(weight)
        else:
            new_state_dict = OrderedDict()
            for k, v in weight.items():
                name = k
                new_state_dict[name] = v
            seg_model.load_state_dict(new_state_dict)

    @staticmethod
    def __pre_to_img(pre):
        result = pre.argmax(axis=1)[0]
        row, col = result.shape
        dst = numpy.zeros((row, col, 3), dtype=numpy.uint8)
        for i in range(N_CLASS):
            dst[result == i] = COLOR_MAP[i]
        dst = numpy.array(dst, dtype=numpy.uint8)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        return dst

    # @staticmethod
    # def __pre_to_train_id(pre):
    #     result = pre.argmax(axis=1)[0]
    #     row, col = result.shape
    #     dst = numpy.zeros((row, col, 1), dtype=numpy.uint8)
    #     for i in range(N_CLASS):
    #         dst[result == i] = COLOR_MAP[i]
    #     dst = numpy.array(dst, dtype=numpy.uint8)
    #     return dst