from PIL import Image
import numpy as np
import os

def boundary(raw_input):

    instance_mask = Image.open(raw_input)

    width = instance_mask.size[0]
    height = instance_mask.size[1]
    mask_array = np.array(instance_mask)

    boundary_mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height -1):
        for j in range(1, width -1):
            if mask_array[i, j] != mask_array[i - 1, j] \
                    or mask_array[i, j] != mask_array[i + 1, j] \
                    or mask_array[i, j] != mask_array[i, j - 1] \
                    or mask_array[i, j] != mask_array[i, j + 1]:
                boundary_mask[i, j] = 255

    boundary_image = Image.fromarray(np.uint8(boundary_mask))

    return boundary_image


if __name__ == '__main__':
    base_path = '/home/mk/Semantic_Segmentation/Seg_dataset/Cityscapes_dataset'
    gt = 'gtFine_trainvaltest/gtFine'
    mode = 'val'

    path = os.path.join(base_path, gt, mode)
    for city_name in os.listdir(path):
        gt_imgs = os.listdir(os.path.join(path, city_name))
        for instance_img in sorted(gt_imgs):
            if '_instanceIds' in instance_img:
                boundary_img = boundary(os.path.join(path, city_name, instance_img))
                save_name = instance_img.split('_')[:4]
                save_name.insert(4, 'boundary.png')

                if not os.path.exists(os.path.join(base_path, 'boundary_image/val', city_name)):
                    os.makedirs(os.path.join(base_path, 'boundary_image/val', city_name))
                boundary_img.save(os.path.join(base_path, 'boundary_image/val', city_name, "_".join(save_name)))
                print('creating boundary image ...', instance_img)