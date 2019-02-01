import os
import cv2

# dir = '/home/mk/munster_rawsegdispopt'
segmentation_dir = '/home/mk/Semantic_Segmentation/DenseASPP-master/Results/Densenet121_prediction_image/DenseNet121/Test_v1_05_2'

# filenames = sorted(os.listdir(dir))
seg_filenames = sorted(os.listdir(segmentation_dir))

for idx in range(len(seg_filenames)):
    image = cv2.imread(os.path.join(segmentation_dir, seg_filenames[idx]))
    image = image[:800,:,:]
    cv2.imwrite('/home/mk/Semantic_Segmentation/DenseASPP-master/Results/Densenet121_prediction_image/DenseNet121/Crop_Test_v01_5_2/{}_crop.png'.format(seg_filenames[idx][:-4]), image)
    print(idx)
# for idx in range(len(filenames)):
#     image = cv2.imread(os.path.join(dir, filenames[idx]))
#     if 'disp' not in filenames[idx]:
#         height, width, channel = image.shape
#         margin_h = int(0.05 * height)
#         margin_w = int(0.05 * width)
#         if channel > 1:
#             image = image[margin_h:height-margin_h, margin_w:width-margin_w, :]
#         else:
#             image = image[margin_h:height - margin_h, margin_w:width - margin_w]
#
#     print(idx)
#     cv2.imwrite('/home/mk/Crop_Img/{}_crop.png'.format(filenames[idx][:-4]), image)