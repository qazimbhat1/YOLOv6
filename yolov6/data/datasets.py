#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
from io import UnsupportedOperation
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path

from multiprocessing.pool import Pool

import cv2
import numpy as np
from tqdm import tqdm
from PIL import ExifTags, Image, ImageOps

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
)
from yolov6.utils.events import LOGGER


# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

import torch
import torchvision.transforms.functional as t_F


class RandomHorizontalFlipWithBbox(torch.nn.Module):
    """Horizontally flip the given image and its bounding box coordinates randomly with a given probability.
    If the image is a torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, labels, status):
        """
        Args:
            img (PIL Image or np.array): Image to be flipped.
            labels (np.array): Bounding box labels in the format [[class, x_min, y_min, x_max, y_max], ...].
            status (bool): Status indicating whether to apply the flip.

        Returns:
            PIL Image or np.array: Randomly flipped image.
            np.array: Updated bounding box labels after flip.
            bool: Updated status after flip.
        """

        
        if status is not None:
            if status == True:
                if isinstance(img, Image.Image):
                    img = t_F.hflip(img)
                else:
                    img = np.flip(img, axis=1)
                flipped_labels = labels.copy()
                flipped_labels[:, 1] = 1.0 - labels[:, 1]
                return img, flipped_labels, status
            else:
                return img, labels, status
        else:
            status = False
            if torch.rand(1) < self.p:
                if isinstance(img, Image.Image):
                    img = t_F.hflip(img)
                else:
                    # print("==========", img)
                    img = np.flip(img, axis=1)
                    # print("+++++++++", img)

                status = True
                flipped_labels = labels.copy()
                flipped_labels[:, 1] = 1.0 - labels[:, 1]
                return img, flipped_labels, status
            return img, labels, status


class TrainValDataset(Dataset):
    '''YOLOv6 train_loader/val_loader, loads images and labels for training and validation.'''
    def __init__(
        self,
        img_dir,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=None,
        task="train",
        specific_shape = False,
        height=1088,
        width=1920,
        gsl=True

    ):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.class_names = data_dict["names"]
        self.img_paths, self.labels = self.get_imgs_labels(self.img_dir)
        self.rect = rect
        self.specific_shape = specific_shape
        self.target_height = height
        self.target_width = width
        self.gsl = gsl
        if self.rect:
            shapes = [self.img_info[p]["shape"] for p in self.img_paths]
            self.shapes = np.array(shapes, dtype=np.float64)
            if dist.is_initialized():
                # in DDP mode, we need to make sure all images within batch_size * gpu_num
                # will resized and padded to same shape.
                sample_batch_size = self.batch_size * dist.get_world_size()
            else:
                sample_batch_size = self.batch_size
            self.batch_indices = np.floor(
                np.arange(len(shapes)) / sample_batch_size
            ).astype(
                np.int_
            )  # batch indices of each image

            self.sort_files_shapes()

        t2 = time.time()
        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (t2 - t1))
        # self.augment = True
        self.hyp = dict(
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.9,
            shear=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )
        # if self.hyp!=None:
        #     self.hyp = hyp
        self.i = 0
    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)

    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        if self.task == 'Train':
            save_dir = "/l/users/mohammad.bhat/FKD_train_full" #save_final_L
            save_path = os.path.join(save_dir,str(self.img_paths[index].split('/')[-1].split('.')[0]))
            state = torch.load(save_path)#, map_location=torch.device('cpu'))
            status, affine_params, outputs = state
            t_feats, t_pred_scores, t_pred_distri = outputs[0], outputs[-2], outputs[-1]
            # outputs = { 'B': t_pred_scores, 'C': t_pred_distri} #'A': t_feats,
            # outputs = [(t_feats, t_pred_scores, t_pred_distri)]
            # for f in t_feats:
            #     print(f.shape)
            # print(t_pred_distri.shape, t_pred_scores.shape)


            # #Method 2
            # save_dir = "/home/projects1_metropolis/tmp/zaid/yolo/save_final_L_numpy"
            # save_path = os.path.join(save_dir,str(self.img_paths[index].split('/')[-1].split('.')[0]))

            # state = torch.load(save_path)#, map_location=torch.device('cpu'))
            # status, affine_params = state

            # import blosc
            # # import lz4
            # # import zstd
            # # import time
            # # start_time = time.time()
            # output_0 = []
            # output = []
            # for i in range(3):
            #     with open(save_path+str(i)+'.blosc', 'rb') as f:
            #         compressed_data = f.read()
            #     decompressed_data = np.frombuffer(blosc.decompress(compressed_data), dtype=np.float32)
            #     # print(output[0][i].dtype)
            #     output_0.append(decompressed_data)
            # output_0[0] = output_0[0].reshape((1,-1, 80, 80))
            # output_0[1] = output_0[1].reshape((1,-1, 40, 40))
            # output_0[2] = output_0[2].reshape((1,-1, 20, 20))

            # for i in range(1,3):
            #     with open(save_path+str(i+2)+'.blosc', 'rb') as f:
            #         compressed_data = f.read()
            #     decompressed_data = np.frombuffer(blosc.decompress(compressed_data), dtype=np.float32)
            #     # print(output[i].dtype) #float32
            #     output.append(decompressed_data)
            
            # t_pred_scores = output[0].reshape((1, -1, 80))
            # t_pred_distri = output[1].reshape((1, -1, 68))
            # # print(t_pred_scores.shape, t_pred_distri.shape)
            # # print(output_0[0].shape, output_0[1].shape, output_0[2].shape)
            # t_feats = output_0
            # # end_time = time.time()
            # # load_time = end_time - start_time
            # # print(f"Time taken to load state blosc: {load_time:.4f} seconds")


            t_pred_scores = torch.from_numpy(t_pred_scores.copy())
            t_pred_distri = torch.from_numpy(t_pred_distri.copy())
            # print(t_pred_scores.shape, t_pred_distri.shape)
            t_feats_dict = {}
            for i in range(len(t_feats)):
                # print(t_feats[i].shape)
                t_feats_dict[str(i)] = torch.from_numpy(t_feats[i].copy())
            t_feats = t_feats_dict
            # raise
        else:
            affine_params = None
            status = [None, None, None]
        self.augment = False
        target_shape = (
                (self.target_height, self.target_width) if self.specific_shape else
                self.batch_shapes[self.batch_indices[index]] if self.rect
                else self.img_size
                )

        # Mosaic Augmentation
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index, target_shape)
            shapes = None

            # MixUp augmentation
            if random.random() < self.hyp["mixup"]:
                img_other, labels_other = self.get_mosaic(
                    random.randint(0, len(self.img_paths) - 1), target_shape
                )
                img, labels = mixup(img, labels, img_other, labels_other)

        else:
            # Load image
            if self.hyp and "shrink_size" in self.hyp:
                # print('h')
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["shrink_size"])
            else:
                img, (h0, w0), (h, w) = self.load_image(index)

            
                
            img, ratio, pad = letterbox(img, target_shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:
                w *= ratio
                h *= ratio
                # new boxes
                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # top left x
                boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # top left y
                boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # bottom right x
                boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # bottom right y
                labels[:, 1:] = boxes

            if self.augment or self.gsl:
                img, labels, affine_params = random_affine(
                    img,
                    labels,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"],
                    new_shape=target_shape,
                    affine_params=affine_params,
                )
            #     print("---",labels.shape)

        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes
        if self.augment or self.gsl:
            img, labels, status = self.general_augment(img, labels, status[0], status[1], status[2])

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # Apply the RandomHorizontalFlipWithBbox transformation here
        # status = None  # Set the initial status to None
        # img, labels, status = RandomHorizontalFlipWithBbox()(img, labels, status)
        # raise
        # if self.task == 'Train':
        #     return torch.from_numpy(img), labels_out, self.img_paths[index], shapes, outputs, None
        if self.task=='Train':
            return torch.from_numpy(img), labels_out, self.img_paths[index], shapes, status, affine_params, t_feats, t_pred_scores, t_pred_distri #outputs

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes, status, affine_params, None, None, None

    def load_image(self, index, shrink_size=None):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        path = self.img_paths[index]
        try:
            im = cv2.imread(path)
            assert im is not None, f"opencv cannot read image correctly or {path} not exists"
        except:
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        if self.specific_shape:
            # keep ratio resize
            ratio = min(self.target_width / w0, self.target_height / h0)

        elif shrink_size:
            ratio = (self.img_size - shrink_size) / max(h0, w0)

        else:
            ratio = self.img_size / max(h0, w0)

        if ratio != 1:
                im = cv2.resize(
                    im,
                    (int(w0 * ratio), int(h0 * ratio)),
                    interpolation=cv2.INTER_AREA
                    if ratio < 1 and not self.augment
                    else cv2.INTER_LINEAR,
                )
        return im, (h0, w0), im.shape[:2]

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes, status, affine_params, t_feats, t_pred_scores, t_pred_distri = zip(*batch)
        # print(img.shape, t_pred_scores.shape)
        if t_feats[0]!=None:
            a_batch = torch.cat([sample['0'] for sample in t_feats], dim=0).squeeze(1)
            b_batch = torch.cat([sample['1'] for sample in t_feats], dim=0).squeeze(1)
            c_batch = torch.cat([sample['2'] for sample in t_feats], dim=0).squeeze(1)
            # t_feats = {'a': a_batch, 'b': b_batch, 'c': c_batch}
            t_feats = [a_batch,b_batch,c_batch]
            t_pred_scores = torch.stack(t_pred_scores, 0).squeeze(1)
            t_pred_distri = torch.stack(t_pred_distri,0).squeeze(1)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, status, affine_params, t_feats, t_pred_scores, t_pred_distri

    def get_imgs_labels(self, img_dirs):
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]
        # we store the cache img file in the first directory of img_dirs
        valid_img_record = osp.join(
            osp.dirname(img_dirs[0]), "." + osp.basename(img_dirs[0]) + "_cache.json"
        )
        NUM_THREADS = min(8, os.cpu_count())
        img_paths = []
        for img_dir in img_dirs:
            assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
            img_paths += glob.glob(osp.join(img_dir, "**/*"), recursive=True)

        img_paths = sorted(
            p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)
        )

        assert img_paths, f"No images found in {img_dir}."
        img_hash = self.get_hash(img_paths)
        LOGGER.info(f'img record infomation path is:{valid_img_record}')
        if osp.exists(valid_img_record):
            with open(valid_img_record, "r") as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                    img_info = cache_info["information"]
                else:
                    self.check_images = True
        else:
            self.check_images = True

        # check images
        if self.check_images and self.main_process:
            img_info = {}
            nc, msgs = 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(
                    pool.imap(TrainValDataset.check_image, img_paths),
                    total=len(img_paths),
                )
                for img_path, shape_per_img, nc_per_img, msg in pbar:
                    if nc_per_img == 0:  # not corrupted
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))

            cache_info = {"information": img_info, "image_hash": img_hash}
            # save valid image paths.
            with open(valid_img_record, "w") as f:
                json.dump(cache_info, f)

        # check and load anns

        img_paths = list(img_info.keys())
        label_paths = img2label_paths(img_paths)
        assert label_paths, f"No labels found."
        label_hash = self.get_hash(label_paths)
        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
            self.check_labels = True

        if self.check_labels:
            cache_info["label_hash"] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths)) if self.main_process else pbar
                for (
                    img_path,
                    labels_per_file,
                    nc_per_file,
                    nm_per_file,
                    nf_per_file,
                    ne_per_file,
                    msg,
                ) in pbar:
                    if nc_per_file == 0:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    if self.main_process:
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            if self.main_process:
                pbar.close()
                with open(valid_img_record, "w") as f:
                    json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            if nf == 0:
                LOGGER.warning(
                    f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                )

        if self.task.lower() == "val":
            if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dirs[0])), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dirs[0]) + ".json"
                )
                TrainValDataset.generate_coco_format_labels(
                    img_info, self.class_names, save_path
                )

        img_paths, labels = list(
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)
                        if info["labels"]
                        else np.zeros((0, 5), dtype=np.float32),
                    )
                    for img_path, info in img_info.items()
                ]
            )
        )
        self.img_info = img_info
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
        )
        return img_paths, labels

    def get_mosaic(self, index, shape):
        """Gets images and labels after mosaic augments"""
        indices = [index] + random.choices(
            range(0, len(self.img_paths)), k=3
        )  # 3 additional image indices
        random.shuffle(indices)
        imgs, hs, ws, labels = [], [], [], []
        for index in indices:
            img, _, (h, w) = self.load_image(index)
            labels_per_img = self.labels[index]
            imgs.append(img)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_img)
        img, labels = mosaic_augmentation(shape, imgs, hs, ws, labels, self.hyp, self.specific_shape, self.target_height, self.target_width)
        return img, labels

    def general_augment(self, img, labels, status_lr=None, status_ud=None, hsv_params=None):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)
        # print(status_lr, status_ud, hsv_params)
        # raise
        # HSV color-space
        # hsv_params = None
        
        # raise
        img, hsv_params = augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
            hsv_params=hsv_params
        )
        
        # raise

        # Flip up-down
        if status_ud==True or (random.random() < self.hyp["flipud"] and status_ud == None):
            img = np.flipud(img)
            status_ud = True
            if nl:
                labels[:, 2] = 1 - labels[:, 2]
        else:
            status_ud = False

        # Flip left-right
        if status_lr==True or (random.random() < self.hyp["fliplr"] and status_lr == None):
            img = np.fliplr(img)
            status_lr = True
            if nl:
                labels[:, 1] = 1 - labels[:, 1]
        else:
            status_lr = False



        # # status_lr = None
        # if status_lr==True or (random.random() < self.hyp["fliplr"] and status_lr == None):
        #     # print("flip image")
        #     # print(type(img))
        #     import matplotlib.pyplot as plt
        #     import matplotlib.patches as patches

            
        #     img = np.fliplr(img)
        #     print(img.shape)
        #     # print(type(img))
        #     # Display the image using matplotlib
        #     # print(img.shape)
        #     # plt.imsave('/home/zbhat/yolo/yolo_orig/yolo_orig_gsl/output_pred/'+str(self.i)+'output_flip.png', img)
        #     # self.i+=1
        #     # plt.imshow(img)
        #     # plt.axis('off')  # Turn off axis labels and ticks
        #     # plt.show()

        #     # Save the image to a file
        #     # plt.imsave('/home/zbhat/yolo/yolo_orig/yolo_orig_gsl/output_image1.png', img)
        #     # raise
        #     status_lr = True
        #     if nl:
        #         print("flip label")
        #         print("1:", labels)
        #         labels[:, 1] = 1 - labels[:, 1]
        #         print("2:", labels)
            
        #     # Load image
        #     image_path = '/home/zbhat/yolo/yolo_orig/yolo_orig_gsl/output_pred/'+str(self.i)+'output_img_plt.jpg'
        #     # plt.imsave(image_path, img)
        #     print(image_path)
        #     cv2.imwrite(image_path, img)
        #     image = cv2.imread(image_path)
        #     print("CV2: ", image)
        #     # if type(image)!=None:
        #     #     print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        #     #     raise
        #     # print("-----------++++++++++++++",image)
        #     # image = plt.imread(image_path)
        #     # image = img
        #     # Bounding box labels
        #     bbox_labels = labels

        #     # Display image with bounding boxes
        #     # plt.imshow(image)

        #     # Draw bounding boxes on the image
        #     for label in bbox_labels:
        #         class_label, x_center, y_center, width, height = label
        #         # print("111111111111111111111111",label, image)
        #         # print(label, image.shape)
        #         x = int(x_center * image.shape[1])
        #         y = int(y_center * image.shape[0])
        #         bbox_width = int(width * image.shape[1])
        #         bbox_height = int(height * image.shape[0])
                
        #         x1 = x - bbox_width // 2
        #         y1 = y - bbox_height // 2
        #         x2 = x1 + bbox_width
        #         y2 = y1 + bbox_height
                
        #         color = (0, 0, 255)  # Red color for bounding box
        #         thickness = 2  # Line thickness
                
        #         cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        #         cv2.putText(image, f'Class {int(class_label)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        #         break

        #     # Save the image with bounding box annotations
        #     output_image_path = '/home/zbhat/yolo/yolo_orig/yolo_orig_gsl/output_pred/'+str(self.i)+'output_overlay.png'
        #     # self.i+=1
        #     # plt.savefig(output_image_path)
        #     # plt.show()
        #     cv2.imwrite(output_image_path, image)

        # else:
        #     status_lr = False

        return img, labels, [status_lr, status_ud, hsv_params]

    def sort_files_shapes(self):
        '''Sort by aspect ratio.'''
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes  # [height, width]
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_paths = [self.img_paths[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):
            ari = ar[self.batch_indices == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [1, maxi]
            elif mini > 1:
                shapes[i] = [1 / mini, 1]
        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
                np.int_
            )
            * self.stride
        )

    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            im = Image.open(im_file)  # need to reload the image after using verify()
            shape = (im.height, im.width)  # (height, width)
            try:
                im_exif = im._getexif()
                if im_exif and ORIENTATION in im_exif:
                    rotation = im_exif[ORIENTATION]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])
            except:
                im_exif = None

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_h, img_w = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()


class LoadData:
    def __init__(self, path, webcam, webcam_addr):
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        if webcam: # if use web camera
            imgp = []
            vidp = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]
        else:
            p = str(Path(path).resolve())  # os-agnostic absolute path
            if os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            else:
                raise FileNotFoundError(f'Invalid path {p}')
            imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
            vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = 'image'
        if len(vidp) > 0:
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None

    # @staticmethod
    def checkext(self, path):
        if self.webcam:
            file_type = 'video'
        else:
            file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'
        return file_type

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.checkext(path) == 'video':
            self.type = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
        return img, path, self.cap

    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
