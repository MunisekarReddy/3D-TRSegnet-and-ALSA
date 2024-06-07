import numpy as np
import warnings
from prettytable import PrettyTable
import cv2 as cv

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA-ALSA', 'AGTO-ALSA', 'POA-ALSA', 'COA-ALSA', 'IPCOS-ALSA']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[0, j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Dataset', 0 + 1, 'Statistical Report ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(50)
    Conv_Graph = Fitness[0]

    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=10,
             label='AOA-ALSA')
    plt.plot(length, Conv_Graph[1, :], color='#89fe05', linewidth=3, marker='*', markerfacecolor='green',
             markersize=10, label='AGTO-ALSA')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
             markersize=10, label='POA-ALSA')
    plt.plot(length, Conv_Graph[3, :], color='#ffff14', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=10, label='COA-ALSA')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=10, label='IPCOS-3D-TRSegnet-ALSA')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    plt.show()


def plot_Images_vs_terms():
    eval1 = np.load('Eval_all_img.npy', allow_pickle=True)
    Terms = ['SSIM', 'Mutual Information', 'RMSE', 'Correlation Coefficient']
    Graph_Terms = [0, 1, 2, 3]
    Algorithm = ['TERMS', 'AOA-ALSA', 'AGTO-ALSA', 'POA-ALSA', 'COA-ALSA', 'IPCOS-ALSA']
    Classifier = ['TERMS', 'CNN', 'U-RSNet', '3DRCNN', 'RSegNet', '3D-TRSegnet']

    value1 = eval1[0, 3, :, :]
    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value1[j, :])
    print('-------------------------------------------------- Dataset', 0 + 1,
          'Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval1.shape[1], eval1.shape[2] + 1))
        for k in range(eval1.shape[1]):
            for l in range(eval1.shape[2]):
                if j == 5:
                    Graph[k, l] = eval1[0, k, l, Graph_Terms[j]]
                else:
                    Graph[k, l] = eval1[0, k, l, Graph_Terms[j]]

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 0], color='#ff000d', width=0.10, edgecolor='k', label="CNN")
        ax.bar(X + 0.10, Graph[:, 1], color='#ffff84', width=0.10, edgecolor='k', label="U-RSNet")
        ax.bar(X + 0.20, Graph[:, 2], color='#87ae73', width=0.10, edgecolor='k', label="3DRCNN")
        ax.bar(X + 0.30, Graph[:, 3], color='#ffb7ce', width=0.10, edgecolor='k', label="RSegNet")
        ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, edgecolor='k', label="3D-TRSegnet")
        plt.xticks(X + 0.10, ('1', '2', '3', '4', '5'))
        plt.xlabel('Images')
        plt.ylabel(Terms[Graph_Terms[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        plt.tight_layout()
        path = "./Results/%s_bar_lrean.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def plot_results_Seg():
    Eval_all = np.load('Eval_all_Seg_.npy', allow_pickle=True)

    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Full = ['TERMS', 'AOA-ALSA', 'AGTO-ALSA', 'POA-ALSA', 'COA-ALSA', 'IPCOS-ALSA', 'CNN', 'U-RSNet', '3DRCNN', 'RSegNet', '3D-TRSegnet']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i]) * 100
                    stats[i, j, 1] = np.min(value_all[j][:, i]) * 100
                    stats[i, j, 2] = np.mean(value_all[j][:, i]) * 100
                    stats[i, j, 3] = np.median(value_all[j][:, i]) * 100
                    stats[i, j, 4] = np.std(value_all[j][:, i]) * 100

            stats[i, 9, :] = stats[i, 4, :]
            Table = PrettyTable()
            Table.add_column(Full[0], Statistics)
            for k in range(10):
                Table.add_column(Full[k + 1], stats[i, k, :])
            print('-------------------------------------------------- Comparison', Terms[i-4],
                  '--------------------------------------------------')
            print(Table)

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 0, :], color='#ff028d', edgecolor='k', width=0.10, label="AOA-ALSA")  # r
            ax.bar(X + 0.10, stats[i, 1, :], color='#0cff0c', edgecolor='k', width=0.10, label="AGTO-ALSA")  # g
            ax.bar(X + 0.20, stats[i, 2, :], color='#0165fc', edgecolor='k', width=0.10, label="POA-ALSA")  # b
            ax.bar(X + 0.30, stats[i, 3, :], color='#fd411e', edgecolor='k', width=0.10, label="COA-ALSA")  # m
            ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="IPCOS-3D-TRSegnet-ALSA")  # k
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Segmentation_%s_%s_alg.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.10, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color='#fc2647', edgecolor='k', width=0.10, label="CNN")
            ax.bar(X + 0.10, stats[i, 6, :], color='#2ee8bb', edgecolor='k', width=0.10, label="U-RSNet")
            ax.bar(X + 0.20, stats[i, 7, :], color='#aa23ff', edgecolor='k', width=0.10, label="3DRCNN")
            ax.bar(X + 0.30, stats[i, 8, :], color='#fe46a5', edgecolor='k', width=0.10, label="RSegNet")
            ax.bar(X + 0.40, stats[i, 4, :] + 0.05, color='k', edgecolor='k', width=0.10,
                   label="IPCOS-3D-TRSegnet-ALSA")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Segmentation_%s_%s_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Images_Sample():
    ct = 1
    if ct == 1:
        cls = ['Dataset_1  CT']
        Original = np.load('CT_Images.npy', allow_pickle=True)
        for i in range(1):
            Orig_1 = Original[i]
            Orig_2 = Original[i + 1]
            Orig_3 = Original[i + 2]
            Orig_4 = Original[i + 3]
            Orig_5 = Original[i + 4]
            Orig_6 = Original[i + 5]
            plt.suptitle('Sample Images from ' + cls[0] + ' ', fontsize=25)
            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig_1)
            plt.subplot(2, 3, 2).axis('off')
            plt.imshow(Orig_2)
            plt.subplot(2, 3, 3).axis('off')
            plt.imshow(Orig_3)
            plt.subplot(2, 3, 4).axis('off')
            plt.imshow(Orig_4)
            plt.subplot(2, 3, 5).axis('off')
            plt.imshow(Orig_5)
            plt.subplot(2, 3, 6).axis('off')
            plt.imshow(Orig_6)
            path = "./Results/Image_results/Sample of _%s_%s_image.png" % (cls[0], i + 1)
            plt.savefig(path)
            plt.show()
            # Sample images
            cv.imwrite('./Results/Sample_Images/%s_Original_image_1.png' % (cls[0]), Orig_1)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_2.png' % (cls[0]), Orig_2)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_3.png' % (cls[0]), Orig_3)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_4.png' % (cls[0]), Orig_4)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_5.png' % (cls[0]), Orig_5)
    mr = 1
    if mr == 1:
        cls = ['Dataset_1 MR']
        Original = np.load('MR_Images.npy', allow_pickle=True)
        for i in range(1):
            Orig_1 = Original[i]
            Orig_2 = Original[i + 1]
            Orig_3 = Original[i + 2]
            Orig_4 = Original[i + 3]
            Orig_5 = Original[i + 4]
            Orig_6 = Original[i + 5]
            plt.suptitle('Sample Images from ' + cls[0] + ' ', fontsize=25)
            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig_1)
            plt.subplot(2, 3, 2).axis('off')
            plt.imshow(Orig_2)
            plt.subplot(2, 3, 3).axis('off')
            plt.imshow(Orig_3)
            plt.subplot(2, 3, 4).axis('off')
            plt.imshow(Orig_4)
            plt.subplot(2, 3, 5).axis('off')
            plt.imshow(Orig_5)
            plt.subplot(2, 3, 6).axis('off')
            plt.imshow(Orig_6)
            path = "./Results/Image_results/Sample of _%s_%s_image.png" % (cls[0], i + 1)
            plt.savefig(path)
            plt.show()
            # Sample images
            cv.imwrite('./Results/Sample_Images/%s_Original_image_1.png' % (cls[0]), Orig_1)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_2.png' % (cls[0]), Orig_2)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_3.png' % (cls[0]), Orig_3)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_4.png' % (cls[0]), Orig_4)
            cv.imwrite('./Results/Sample_Images/%s_Original_image_5.png' % (cls[0]), Orig_5)


def Image_Results():
    CT = np.load('CT_Images.npy')
    MR = np.load('MR_Images.npy')
    Reg = np.load('Registered_image.npy')
    Seg = np.load('segmented_image.npy')
    for i in range(5):
        ct = CT[i]
        mri = MR[i]
        reg = Reg[i]
        seg = Seg[i]

        cv.imshow('Registered image', np.uint8(reg))
        cv.imshow("mri image", np.uint8(mri))
        cv.imshow("ct image", np.uint8(ct))
        cv.imshow("Segmented image", np.uint8(seg))
        cv.waitKey(0)

        cv.imwrite('./Results/Image_Results/CT_image_' + str(i + 1) + '.png', np.uint8(ct))
        cv.imwrite('./Results/Image_Results/MRI_image_' + str(i + 1) + '.png', np.uint8(mri))
        cv.imwrite('./Results/Image_Results/Registered_image_' + str(i + 1) + '.png', np.uint8(reg))
        cv.imwrite('./Results/Image_Results/segmented_image_' + str(i + 1) + '.png', np.uint8(seg))


if __name__ == '__main__':
    plot_conv()
    plot_Images_vs_terms()
    plot_results_Seg()
    Images_Sample()
    Image_Results()
