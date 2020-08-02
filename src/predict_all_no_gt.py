import glob
import os
import shutil

import keras.backend as K
import nibabel as nib
import numpy as np
from EncoderDecoder_brats import NASNetMobile_enc_dec, ResNet50_enc_dec
from datagenerator import get_data_no_gt, refine_unc_maps, refine_labels


def run_model_2d(model, img_flair, img_t1, img_t2, img_t1ce, zp):
    '''
    This function run 2d inference on all slices
    :param model: model object to infer on
    :param img_flair:
    :param img_t1:
    :param img_t2:
    :param img_t1ce:
    :param zp:
    :return: Probability maps for all classes after removing zero padding
    '''
    H, W, SLC = img_t1.shape
    prob_out = np.zeros((H, W, SLC, 4))
    batch_size = 1
    for sl in range(0, SLC - batch_size + 1, batch_size):
        ch0 = np.expand_dims(np.transpose(img_flair[:, :, sl:sl + batch_size], (2, 0, 1)), -1)
        ch1 = np.expand_dims(np.transpose(img_t1[:, :, sl:sl + batch_size], (2, 0, 1)), -1)
        ch2 = np.expand_dims(np.transpose(img_t1ce[:, :, sl:sl + batch_size], (2, 0, 1)), -1)
        ch3 = np.expand_dims(np.transpose(img_t2[:, :, sl:sl + batch_size], (2, 0, 1)), -1)
        x = np.concatenate((ch0, ch1, ch2, ch3), -1)
        y_pred = model.predict(x)
        prob_out[:, :, sl, :] = np.squeeze(y_pred)
    prob_out_crop = np.zeros((zp.r, zp.c, SLC, 4))
    for indx in range(4):
        prob_out_crop[:, :, :, indx] = zp.remove_zpad(prob_out[:, :, :, indx])
    prob_out_crop = prob_out_crop.astype('float32')
    return prob_out_crop


def run_model_2d_all_files(f_dir, model, dir_prob, flag=0, orient=0, count=0):
    for k, fname in enumerate(f_dir):
        print 'Processing: ', k + 1 + count, '/', 5 * num_file
        img_flair, img_t1, img_t2, img_t1ce, zp = get_data_no_gt(fname, flag=flag, orient=orient)
        prob_out = run_model_2d(model, img_flair, img_t1, img_t2, img_t1ce, zp)
        if orient == 0:  # axial
            pass
        if orient == 1:  # coronal
            prob_out = np.transpose(prob_out, (0, 2, 1, 3))
        if orient == 2:  # sagittal
            prob_out = np.transpose(prob_out, (2, 0, 1, 3))
        fname_out = dir_prob + fname.split('/')[-1][:-9] + 'prob.npz'
        np.savez_compressed(fname_out, prob=prob_out)


def save_nifti(fname, dir_out, out_label):
    imgObj = nib.load(fname)
    imgObj_w = nib.Nifti1Image(out_label, imgObj.affine, imgObj.header)
    f_new = os.path.join(dir_out, fname.split('/')[-1][:-10] + '.nii.gz')
    nib.save(imgObj_w, f_new)


def refine_lable4(x, thrshld):
    label4_sum = np.sum(x == 4)
    if label4_sum < thrshld:
        x[x == 4] = 0
    return x


if __name__ == '__main__':
    # --- define paths ----- #
    path_data = '../data/*t1.nii.gz'
    dir_out = '../data/results/'

    W_Resnet50_ax = '../weights/ResNet50_tum_only/axial/model-033.hdf5'
    W_Resnet50_cor = '../weights/ResNet50_tum_only/cor/model-034.hdf5'
    W_Resnet50_sag = '../weights/ResNet50_tum_only/sag/model-027.hdf5'
    W_NasnetMob_ax = '../weights/NASNetMobile/axial/model-023.hdf5'
    W_NasnetMob_cor = '../weights/NASNetMobile/cor/model-021.hdf5'

    dir_prob_res_ax = '../scratch/Resnet50_ax/'
    dir_prob_res_cor = '../scratch/Resnet50_cor/'
    dir_prob_res_sag = '../scratch/Resnet50_sag/'
    dir_prob_nas_ax = '../scratch/NasnetMob_ax/'
    dir_prob_nas_cor = '../scratch/NasnetMob_cor/'

    if not os.path.exists(dir_prob_res_ax):
        os.makedirs(dir_prob_res_ax)
    if not os.path.exists(dir_prob_res_cor):
        os.makedirs(dir_prob_res_cor)
    if not os.path.exists(dir_prob_res_sag):
        os.makedirs(dir_prob_res_sag)
    if not os.path.exists(dir_prob_nas_ax):
        os.makedirs(dir_prob_nas_ax)
    if not os.path.exists(dir_prob_nas_cor):
        os.makedirs(dir_prob_nas_cor)

    # --------- create list of files -------------- #
    f_dir = glob.glob(path_data)
    num_file = len(f_dir)
    assert (num_file > 0), "No files in the folder"

    # # ------- Load Model Definition Resnet -------- #
    model = ResNet50_enc_dec(H=256, W=256, weights=None, ninchannel=4, noutchannel=4, isregression=False,
                             ismuticlass=True)
    # ----- Run 2d Resnet Network Axial ----- #
    model.load_weights(W_Resnet50_ax)
    run_model_2d_all_files(f_dir, model, dir_prob_res_ax, orient=0, count=0 * num_file)

    # ----- Run 2d Resnet Network coronal ----- #
    model.load_weights(W_Resnet50_cor)
    run_model_2d_all_files(f_dir, model, dir_prob_res_cor, orient=1, count=1 * num_file)

    # ----- Run 2d Resnet Network coronal ----- #
    model.load_weights(W_Resnet50_sag)
    run_model_2d_all_files(f_dir, model, dir_prob_res_sag, orient=2, count=2 * num_file)

    # # ------- Load Model Definition NasnetMob -------- #
    K.clear_session()
    model = NASNetMobile_enc_dec(H=256, W=256, weights=None, ninchannel=4, noutchannel=4, isregression=False,
                                 ismuticlass=True)
    #
    # ----- Run 2d NasnetMob Network Axial ----- #
    model.load_weights(W_NasnetMob_ax)
    run_model_2d_all_files(f_dir, model, dir_prob_nas_ax, orient=0, count=3 * num_file)

    # ----- Run 2d NasnetMob Network coronal ----- #
    model.load_weights(W_NasnetMob_cor)
    run_model_2d_all_files(f_dir, model, dir_prob_nas_cor, orient=1, count=4 * num_file)

    # ----- Run ensemble ----- #
    for k, fname in enumerate(f_dir):
        print 'Generating lables for : ', fname
        fname_res_ax = dir_prob_res_ax + fname.split('/')[-1][:-9] + 'prob.npz'
        prob_res_ax = np.load(fname_res_ax)['prob']
        fname_res_cor = dir_prob_res_cor + fname.split('/')[-1][:-9] + 'prob.npz'
        prob_res_cor = np.load(fname_res_cor)['prob']
        fname_res_sag = dir_prob_res_sag + fname.split('/')[-1][:-9] + 'prob.npz'
        prob_res_sag = np.load(fname_res_sag)['prob']

        fname_nas_ax = dir_prob_nas_ax + fname.split('/')[-1][:-9] + 'prob.npz'
        prob_nas_ax = np.load(fname_nas_ax)['prob']
        fname_nas_cor = dir_prob_nas_cor + fname.split('/')[-1][:-9] + 'prob.npz'
        prob_nas_cor = np.load(fname_nas_cor)['prob']

        prob_out = (prob_res_ax + prob_res_cor + prob_res_sag + prob_nas_ax + prob_nas_cor) / 5.0
        y_seg = np.argmax(prob_out, axis=-1)
        y_seg = refine_labels(y_seg)
        y_seg[y_seg == 3] = 4
        y_seg = refine_lable4(y_seg, thrshld=500)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        save_nifti(fname, dir_out, y_seg)
    # ----- delete temporary files ---- #
    shutil.rmtree('../scratch/')

