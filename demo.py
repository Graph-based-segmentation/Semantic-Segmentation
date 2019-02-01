import argparse
from inference_boundary import Inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DenseASPP inference code.')
    parser.add_argument('--model_name', default='DenseASPP121', help='segmentation model')
    parser.add_argument('--model_path', default='/home/mk/Semantic_Segmentation/ckpt/Model/ImageNet/DenseNet121_v3/segImageNet_v01_0/model-8283987.pkl',
                        help='weight path')
    parser.add_argument('--img_dir', default='/home/mk/Semantic_Segmentation/Seg_dataset/Cityscapes_dataset/leftImg8bit_trainvaltest/leftImg8bit/val',
                        help='image dir')
    parser.add_argument('--boundary_dir', default='/home/mk/Semantic_Segmentation/Seg_dataset/Cityscapes_dataset/boundary_image')
    args = parser.parse_args()

    infer = Inference(args.model_name, args.model_path)
    infer.folder_inference(args.img_dir, args.boundary_dir, is_multiscale=False)
