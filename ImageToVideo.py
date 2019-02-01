import cv2
import os

img_dir = '/home/mk/Semantic_Segmentation/DenseASPP-master/Results/Densenet121_prediction_image/DenseNet121/Hanyang_v01_3'
raw_img_dir = '/home/mk/Semantic_Segmentation/SegVideo/Hanyang_20190108_2'
video_dir = '/home/mk/HeLp_challenge/Sample_data/output_image'

MODE = 'NULL'

images = [img for img in sorted(os.listdir(img_dir))]
raw_images = [raw_img for raw_img in sorted(os.listdir(raw_img_dir))]

frame = cv2.imread(os.path.join(img_dir, images[0]))
height, width, channel = frame.shape

if MODE == 'VIDEO':
    v_con_dir = '/home/mk/HeLp_challenge/Sample_data/output_image'
    v_images = [v_imgs for v_imgs in sorted(os.listdir(v_con_dir))]

    video = cv2.VideoWriter('/home/mk/HeLp_challenge/Sample_data/output_image/brain_tumor.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                            10.0, (width, height), True)

    for idx, v_img in enumerate(v_images):
        video.write(cv2.imread(os.path.join(v_con_dir, v_img)))
        print('converting!: {}'.format(idx + 1))
    video.release()

elif MODE == 'NULL':
    i=0
    for idx in range(len(images)):
        seg_img = cv2.imread(os.path.join(img_dir, images[idx]))
        raw_img = cv2.imread(os.path.join(raw_img_dir, raw_images[idx]))
        h_img = cv2.vconcat([raw_img, seg_img])

        cv2.imwrite('/home/mk/Semantic_Segmentation/DenseASPP-master/Results/Hanyang_v01_3/{:06d}.png'.format(i), h_img)
        i+=1

        print('converting!{}'.format(idx + 1))

# for idx in range(len(images)):
#     h_images = cv2.hconcat(images, raw_images)

# h_images = [cv2.hconcat(images, raw_images) for idx in range(len(images))]

# video = cv2.VideoWriter('/home/mk/Semantic_Segmentation/SegVideo/raw_seg.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

# for h_img in h_images:
#     video.write(cv2.imread(os.path.join(img_dir, h_img)))
#     print('converting!{}'.format(h_img))
# video.release()