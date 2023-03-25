from pathlib import Path

import tensorflow as tf
import DamDataGenerator as data
# import focalloss_funcs as floss
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os
import sklearn.metrics as skm

loss_dir = './output/nets/losses/'
fv_dir = './output/nets/firstvals/'

model_dir = './output/nets/'
fig_dir = './output/figs/'


def plot_training(model_output_dir: Path):
    train_loss = []
    val_loss = []

    losses_path = model_output_dir / "losses.csv"
    with open(losses_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            losses = r['epoch;loss;val_loss'].split(';')
            train_loss.append(float(losses[1]))
            val_loss.append(float(losses[2]))

    first_val_path = model_output_dir / "firstval.npy"
    first_vals = np.load(str(first_val_path.absolute()))
    plt.title(str(first_vals))
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss)
    plt.plot(np.arange(1, len(train_loss) + 1), val_loss)
    plt.legend(['train', 'val'])
    training_losses_path = model_output_dir / 'training_losses.png'
    plt.savefig(training_losses_path)
    plt.close()


def do_perf_analysis(model: tf.keras.Model, model_output_path: Path, batch_size: int = 1200):
    net_name = model_output_path.stem
    if 'Nir' in net_name:
        RGBN = 3
    elif 'Rgb' in net_name:
        RGBN = [0, 1, 2]
    else:
        RGBN = [0, 1, 2, 3]
    # alpha = float(filename.split('_al')[1].split('_')[0])
    # gamma = float(filename.split('_gm')[1].split('_')[0])
    # , custom_objects={ 'binary_focal_loss_fixed': floss.binary_focal_loss(alpha=alpha, gamma=gamma) })

    data_generator = data.DataGenerator(batch_size=batch_size, test=True, RGBN=RGBN)
    inputs, labels = data_generator.get_testbatch()  # #__getitem__(1) #
    loss = model.evaluate(inputs, labels, verbose=0)
    print("Test loss for " + net_name + ' is ' + str(loss))

    outputs = model(inputs, training=False).numpy()
    fpr, tpr, threshold = skm.roc_curve(labels, outputs)
    plt.plot(fpr, tpr)
    plt.title(skm.roc_auc_score(labels, outputs))
    roc_curve_path = model_output_path / f"actual_ROC.png"
    plt.savefig(roc_curve_path)
    plt.cla()
    plt.close()

    pos = np.where(outputs >= .5)[0]
    neg = np.where(outputs < .5)[0]
    cpos = np.where(labels == 1)[0]
    cneg = np.where(labels == 0)[0]
    TP = len(np.intersect1d(pos, cpos))
    TN = len(np.intersect1d(neg, cneg))
    FP = len(np.intersect1d(pos, cneg))
    FN = len(np.intersect1d(neg, cpos))

    print(f"Results for {net_name}")
    print('accuracy: ', (TP + TN) / (TP + FP + TN + FN))
    print('hit rate: ', TP / (TP + FN))
    try:
        print('precision: ', TP / (TP + FP))
    except:
        print('precision: NaN')
    print('selectivity: ', TN / (TN + FP))

    accs = []
    HRs = []
    precs = []
    sels = []
    for thresh in range(0, 105, 5):
        pos = np.where(outputs >= thresh / 100)[0]
        neg = np.where(outputs < thresh / 100)[0]
        TP = len(np.intersect1d(pos, cpos))
        TN = len(np.intersect1d(neg, cneg))
        FP = len(np.intersect1d(pos, cneg))
        FN = len(np.intersect1d(neg, cpos))
        accs.append((TP + TN) / (TP + FP + TN + FN))
        HRs.append(TP / (TP + FN))
        try:
            precs.append(TP / (TP + FP))
        except:
            precs.append(np.nan)
        sels.append(TN / (TN + FP))

    plt.plot(accs, c='k')
    plt.plot(HRs, c='r')
    plt.plot(precs, c='b')
    plt.plot(sels, c='y')
    plt.legend(['Acc', 'TP/(TP+FN)', 'TP/(TP+FP)', 'TN/(TN+FP)'])
    plt.title(net_name)
    plt.xticks(np.arange(0, 21, 10), np.arange(0, 1.5, .5))
    roc_curve_path = model_output_path / "ROC.png"
    plt.savefig(roc_curve_path)
    plt.close()


def do_loc_perf_analysis(net, batch_size=1200):
    filename = [f for f in os.listdir(loss_dir) if net in f][0][:-10]
    if 'Nir' in filename:
        RGBN = 3
    elif 'Rgb' in filename:
        RGBN = [0, 1, 2]
    else:
        RGBN = [0, 1, 2, 3]
    # alpha = float(filename.split('_al')[1].split('_')[0])
    # gamma = float(filename.split('_gm')[1].split('_')[0])
    reconstructed_model = tf.keras.models.load_model(
        model_dir + filename)  # , custom_objects={ 'binary_focal_loss_fixed': floss.binary_focal_loss(alpha=alpha, gamma=gamma) })

    for loc in ['01', '02', '03', '04', '05', '06']:
        DataGen = data.DataGenerator(batch_size=batch_size, test=True, RGBN=RGBN, loc=loc)
        inputs, labels = DataGen.get_testbatch()  # #__getitem__(1) #
        loss = reconstructed_model.evaluate(inputs, labels, verbose=0)
        print('Test loss for loc ' + loc + ' for ' + net + ' is ' + str(loss))

        outputs = reconstructed_model(inputs, training=False).numpy()

        print(loc)
        print('ROC AUC: ', skm.roc_auc_score(labels, outputs))
        print('AP: ', skm.average_precision_score(labels, outputs))

        pos = np.where(outputs >= .5)[0]
        neg = np.where(outputs < .5)[0]
        cpos = np.where(labels == 1)[0]
        cneg = np.where(labels == 0)[0]
        TP = len(np.intersect1d(pos, cpos))
        TN = len(np.intersect1d(neg, cneg))
        FP = len(np.intersect1d(pos, cneg))
        FN = len(np.intersect1d(neg, cpos))

        print(net)
        print('accuracy: ', (TP + TN) / (TP + FP + TN + FN))
        print('hit rate: ', TP / (TP + FN))
        try:
            print('precision: ', TP / (TP + FP))
        except:
            print('precision: NaN')
        print('selectivity: ', TN / (TN + FP))

        accs = []
        HRs = []
        precs = []
        sels = []
        for thresh in range(0, 105, 5):
            pos = np.where(outputs >= thresh / 100)[0]
            neg = np.where(outputs < thresh / 100)[0]
            TP = len(np.intersect1d(pos, cpos))
            TN = len(np.intersect1d(neg, cneg))
            FP = len(np.intersect1d(pos, cneg))
            FN = len(np.intersect1d(neg, cpos))
            accs.append((TP + TN) / (TP + FP + TN + FN))
            HRs.append(TP / (TP + FN))
            try:
                precs.append(TP / (TP + FP))
            except:
                precs.append(np.nan)
            sels.append(TN / (TN + FP))

        plt.subplot(2, 3, int(loc));
        plt.title(loc)
        plt.plot(accs, c='k');
        plt.plot(HRs, c='r');
        plt.plot(precs, c='b');
        plt.plot(sels, c='y')  # plt.legend(['Acc','HR','Prec','Sel']); #plt.title(filename);
        plt.xticks(np.arange(0, 21, 10), np.arange(0, 1.5, .5))
    plt.tight_layout()
    plt.savefig(fig_dir + 'locROC_' + filename.split('_L')[0] + '.png');
    plt.close()
