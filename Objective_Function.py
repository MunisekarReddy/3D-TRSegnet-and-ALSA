import numpy as np
from Evaluation import net_evaluation
from Global_Vars import Global_Vars
from Model_ALSA import Model_ALSA


def objfun_Segmentation(Soln):
    Feat = Global_Vars.Feat
    CT = Global_Vars.CT
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            orig, predict = Model_ALSA(Feat, CT, sol)
            EVAl = []
            for img in range(len(predict)):
                Eval = net_evaluation(predict[img], CT[img])
                EVAl.append(Eval)
            mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
            Fitn[i] = 1 / mean_EVAl[0, 4]  # Dice Coefficient
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        orig, predict = Model_ALSA(Feat, CT, sol)
        EVAl = []
        for img in range(len(predict)):
            Eval = net_evaluation(predict[img], CT[img])
            EVAl.append(Eval)
        mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / mean_EVAl[0, 4]  # Dice Coefficient
        return Fitn
