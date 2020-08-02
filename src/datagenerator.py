import glob
import nibabel as nib
import numpy as np
from keras import backend as K
from skimage import measure
import keras
import os


def get_data_no_gt(path, flag=0, orient=0):
    zp = image_zpad()
    # --- Read nifti files ---- #
    f_name_flair = path[:-9] + 'flair.nii.gz'
    f_name_t1 = path[:-9] + 't1.nii.gz'
    f_name_t1ce = path[:-9] + 't1ce.nii.gz'
    f_name_t2 = path[:-9] + 't2.nii.gz'
    img_flair = read_nii(f_name_flair, flag)
    img_t1 = read_nii(f_name_t1, flag)
    img_t1ce = read_nii(f_name_t1ce, flag)
    img_t2 = read_nii(f_name_t2, flag)
    if orient == 1:
        img_flair = np.transpose(img_flair, (0, 2, 1))
        img_t1 = np.transpose(img_t1, (0, 2, 1))
        img_t1ce = np.transpose(img_t1ce, (0, 2, 1))
        img_t2 = np.transpose(img_t2, (0, 2, 1))
    if orient == 2:
        img_flair = np.transpose(img_flair, (1, 2, 0))
        img_t1 = np.transpose(img_t1, (1, 2, 0))
        img_t1ce = np.transpose(img_t1ce, (1, 2, 0))
        img_t2 = np.transpose(img_t2, (1, 2, 0))

    img_flair = normalize(zp.zpad_to_256(img_flair))
    img_t1 = normalize(zp.zpad_to_256(img_t1))
    img_t1ce = normalize(zp.zpad_to_256(img_t1ce))
    img_t2 = normalize(zp.zpad_to_256(img_t2))
    return img_flair, img_t1, img_t2, img_t1ce, zp


def refine_unc_maps(prob, thrshld=10.0, ntop=1):
    mask_binary = np.zeros(prob.shape)
    mask = np.zeros(prob.shape)
    mask_binary[prob > thrshld] = 1.0
    all_labels = measure.label(mask_binary, connectivity=3)
    label_indx = find_top_areas_blob(all_labels, top_frac=0.5)
    for k in label_indx:
        mask[all_labels == k] = 1
    prob_wt = prob * mask
    return prob_wt


def refine_labels(label):
    H, W, SL = label.shape

    label_l1 = np.zeros((H, W, SL))
    label_l1[label == 1] = 1
    label_l1 = refine_single_label(label_l1)

    label_l2 = np.zeros((H, W, SL))
    label_l2[label == 2] = 1
    label_l2 = refine_single_label(label_l2)
    label_l2[label_l2 == 1] = 2

    label_l3 = np.zeros((H, W, SL))
    label_l3[label == 3] = 1
    label_l3 = refine_single_label(label_l3)
    label_l3[label_l3 == 1] = 3

    label_out = label_l1 + label_l2 + label_l3
    return label_out


def normalize(x):
    eps = 1e-12
    y = (x - np.mean(x))
    y = y / (eps + np.std(x))
    y[y < -6.0] = -6.0
    y[y > 6.0] = 6.0
    return y


def read_nii(f_name, f_indx=0):
    f_img_obj = nib.load(f_name)
    f_img = np.single(f_img_obj.get_data())
    f_img = augument_data(f_img, f_indx)
    return f_img


def find_top_areas_blob(label, top_frac=0.5):
    N = np.max(label) + 1
    ntop = int(top_frac * N + 0.5)
    areas = np.zeros((N, 1))
    for k in range(0, N):
        areas[k] = np.sum(label == (k + 1))
    indx_sorted = np.flipud(np.argsort(areas, axis=0))
    label_indx = []
    if N >= ntop:
        for k in range(0, ntop):
            if areas[indx_sorted[k]] > 0:
                label_indx.append(indx_sorted[k] + 1)
    return label_indx


def refine_single_label(label):
    # ------ perform connected component analysis ---- #
    label_out = np.zeros(label.shape)
    all_labels = measure.label(label, connectivity=3)
    label_indx = find_top_areas_blob(all_labels)
    for k in label_indx:
        label_out[all_labels == k] = 1
    return label_out


def augument_data(f_img, f_indx):
    if f_indx == 1:
        f_img = np.rot90(f_img, 1, (0, 1))
    elif f_indx == 2:
        f_img = np.rot90(f_img, 2, (0, 1))
    elif f_indx == 3:
        f_img = np.rot90(f_img, 3, (0, 1))
    elif f_indx == 4:
        f_img = np.fliplr(f_img)
    elif f_indx == 5:
        f_img = np.flipud(f_img)
    elif f_indx == 6:
        f_img = np.rot90(np.flipud(f_img), 1, (0, 1))
    elif f_indx == 7:
        f_img = np.rot90(np.flipud(f_img), 3, (0, 1))
    else:
        f_img = f_img
    return f_img


class image_zpad():
    def __init__(self):
        self.r = None
        self.c = None

    def zpad_to_256(self, img):
        self.r, self.c, _ = img.shape
        self.zpad_r = 256 - self.r
        self.zpad_c = 256 - self.c
        self.zpad_r0 = int(np.ceil(self.zpad_r / 2.))
        self.zpad_r1 = int(np.floor(self.zpad_r / 2.))
        self.zpad_c0 = int(np.ceil(self.zpad_c / 2.))
        self.zpad_c1 = int(np.floor(self.zpad_c / 2.))
        img = np.pad(img, ((self.zpad_r0, self.zpad_r1), (self.zpad_c0, self.zpad_c1), (0, 0)), 'constant')
        return img

    def remove_zpad(self, img):
        img = img[self.zpad_r0:self.zpad_r0 + self.r, self.zpad_c0:self.zpad_c0 + self.c, :]
        return img
