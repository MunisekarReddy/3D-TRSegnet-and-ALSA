import os
import numpy as np
import cv2 as cv
from numpy import matlib
from AGTO import AGTO
from AOA import AOA
from COA import COA
from Global_Vars import Global_Vars
from Model_3D_TRegNet import Model_3D_TRegNet
from Model_ALSA import Model_ALSA
from Model_CNN import Model_CNN
from Model_Interactive_segmentation import Model_Interactive_segmentation
from Model_RSegNet import Model_RSegNet
from Model_U_RSNet import Model_U_RSNet
from Objective_Function import objfun_Segmentation
from POA import POA
from Proposed import PROPOSED
from Plot_results import *

# Read the Dataset
an = 0
if an == 1:
    Dataset_fold = './Dataset/Dataset/Patient Data/'
    Dataset_fold_path = os.listdir(Dataset_fold)
    CT = []
    MRI = []
    for i in range(len(Dataset_fold_path)):
        Data_file = Dataset_fold + Dataset_fold_path[i]
        Data_file_path = os.listdir(Data_file)
        for j in range(len(Data_file_path)):
            Img_File = Data_file + '/' + Data_file_path[j]
            Img_File_name = Img_File.split('/')
            if Img_File_name[4] == 'ct.jpg':
                ct_img = cv.imread(Img_File)
                ct_img_resized = cv.resize(ct_img, (256, 256))
                CT.append(ct_img_resized)

            elif Img_File_name[4] == 'mri.jpg':
                mri_img = cv.imread(Img_File)
                mri_img_resized = cv.resize(mri_img, (256, 256))
                MRI.append(mri_img_resized)

            elif Img_File_name[5] == 'fusion.jpg':
                mri_img = cv.imread(Img_File)
                mri_img_resized = cv.resize(mri_img, (256, 256))
                MRI.append(mri_img_resized)

    np.save('CT_Images.npy', CT)
    np.save('MR_Images.npy', MRI)
    np.save('Fused_Image.npy', MRI)


# Image Registration
an = 0
if an == 1:
    image1 = np.load('CT_Images.npy')
    image2 = np.load('MR_Images.npy')
    Registered_Images = []
    for j in range(len(image1)): 
        print(j, 'mr', len(image1))
        img1 = image1[j]
        img2 = image2[j]
        Registered = Model_3D_TRegNet(img1, img2)
        Registered_Images.append(Registered)
    np.save('Registered_Image.npy', Registered_Images)


# optimization for Segmentation
an = 0
if an == 1:
    CT = np.load('CT_Images.npy', allow_pickle=True)
    MR = np.load('MR_Images.npy', allow_pickle=True)
    Feat = np.load('Registered_Image.npy', allow_pickle=True)
    Global_Vars.Feat = Feat
    Global_Vars.CT = CT
    Global_Vars.MR = MR
    Npop = 10
    Chlen = 3 * 3  # Gaussian kernel
    xmin = matlib.repmat(np.asarray([-1.0]), Npop, Chlen)
    xmax = matlib.repmat(np.asarray([1.0]), Npop, Chlen)
    fname = objfun_Segmentation
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("AOA...")
    [bestfit1, fitness1, bestsol1, time1] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("AGTO...")
    [bestfit2, fitness2, bestsol2, time2] = AGTO(initsol, fname, xmin, xmax, Max_iter)  # AGTO

    print("POA...")
    [bestfit4, fitness4, bestsol4, time3] = POA(initsol, fname, xmin, xmax, Max_iter)  # POA

    print("COA...")
    [bestfit3, fitness3, bestsol3, time4] = COA(initsol, fname, xmin, xmax, Max_iter)  # COA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    bestfit = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', bestfit)
    np.save('BestSol_CLS.npy', BestSol_CLS)

# Segmentation
an = 0
if an == 1:
    Data = np.load('Registered_Image.npy', allow_pickle=True)  # Load the Data
    Target = np.load('CT_Images.npy', allow_pickle=True)
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)
    SSL = Model_RSegNet(Data, Target)
    GAN = Model_Interactive_segmentation(Data, Target)
    MPSCL = Model_CNN(Data, Target)
    ASCL = Model_U_RSNet(Data, Target)
    Proposed = Model_ALSA(Data, Target, BestSol[4, :].astype(np.int16))
    Seg = [SSL, GAN, MPSCL, ASCL, Proposed]
    np.save('segmented_image.npy', Proposed)


plot_conv()
plot_Images_vs_terms()
plot_results_Seg()
Images_Sample()
Image_Results()
