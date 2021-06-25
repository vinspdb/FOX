import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import pickle as pk
import os
from sys import argv
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# make a class prediction for one row of data

def read_rule(model):
    r = []
    for varname, members in model.layer['fuzzify'].varmfs.items():
                #r.append('[')
                #r.append('Variable {}'.format(varname))
                for mfname, mfdef in members.mfdefs.items():
                    r.append(varname +' _ {}: {}({})'.format(mfname, mfdef.__class__.__name__, ', '.join(['{}={}'.format(n, p.item()) for n, p in mfdef.named_parameters()])))
                #r.append(']')
    r = np.array(r)
    x = model.layer['consequent'].coeff
    scaler = MinMaxScaler()
    lis = []
    i = 0

    y = Variable(x, requires_grad=True)
    y = y.detach().numpy()

    from scipy.special import softmax
    while i < len(y):
        scaled = softmax(y[i])
        lis.append(scaled)
        i = i+1

    rstr = []

    vardefs = model.layer['fuzzify'].varmfs
    print(vardefs)
    rule_ants = model.layer['rules'].extra_repr(vardefs).split('\n')

    for i, crow in enumerate(lis):
        rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i])+ ' THEN {}'.format(crow.tolist()))
        
    rstr = np.array(rstr)

    ii = 0
    cons = []
    while ii<len(rstr):
        v = (rstr[ii].split('THEN')[1])
        v = v.replace('[','')
        v = v.replace(']','')
        v = list(v.split(','))
        v = [float(i) for i in v]

        cons.append(np.argmax(v))
        ii = ii + 1

    return cons, rstr


def get_fire_strength(model,pred2):
    list_fire_rule = np.load('list_fire_rule.npy')
    regole_anfis, rstr = (read_rule(model))

    iii = 0
    while iii < len(list_fire_rule):
        jjj = 0
        dfr = {}
        while jjj < len(list_fire_rule[iii]):
            dfr[jjj] = list_fire_rule[iii][jjj]
            jjj = jjj + 1
        # print(dfr)

        # dfr = sorted(dfr.items(), key=lambda x:x[1], reverse=True)
        dfr_sort = sorted(dfr, key=dfr.get, reverse=True)
        print(dfr_sort)
        print(dfr)
        # print(dfr_sort)

        z = 0
        #print(conv_test_for_check[iii])
        while z < len(dfr_sort):
            if regole_anfis[dfr_sort[z]] == pred2:
                print("Predicted Value->" + str(pred2))
                print('RULE->' + str(rstr[dfr_sort[z]]))
                rule = str(rstr[dfr_sort[z]])
                print('\n')
                break
            else:
                print('no ok')
                print('RULE->' + str(rstr[dfr_sort[z]]))
                rule = str(rstr[dfr_sort[z]])

            z = z + 1
        iii = iii + 1
    return rule, dfr, dfr_sort


def metrics(dataset_name, columns_sel):
    model = torch.load('models/model_'+dataset_name + '.h5')
    #model = torch.load('G_model_geo_0.h5')
    print(model)
    #exit()

    df_train = pd.read_csv("dataset/"+dataset_name+"/"+dataset_name+'_train.csv', header=0, sep=',')
    df_train = df_train[columns_sel]
    x_train = df_train.values
    x_train = torch.Tensor(x_train)
    #print(x_train)

    #import experimental
    #experimental.plot_all_mfs(model, x_train)
    #model = load_model("dataset/"+dataset_name+"/"+dataset_name+".h5")
    pd_len = pd.read_csv("dataset/"+dataset_name+"/len_test"+dataset_name+".csv", header=0, sep=',')
    max_len = pd_len['Len'].max()

    df_test = pd.read_csv("dataset/"+dataset_name+"/"+dataset_name+'_test.csv', header=0, sep=',')
    df_test = df_test[columns_sel]
    

    image_all = df_test.values
    image_all = image_all[:, 0:len(df_test.columns) - 1]
    #image_all = scaler.transform(image_all)
    #print(image_all)


    y_test = pd.read_csv("dataset/"+dataset_name+"/"+dataset_name+"_test.csv")
    y_test = y_test[y_test.columns[-1]]

    f = open(dataset_name+"_results.csv", "w")
    f.write("SELECTED COLUMNS;"+"\n")
    for element in columns_sel:
        f.write(element + "\n")
    f.write("----------------;"+"\n")

    f.write("LENGHT;NUMBEROFSAMPLES;ROC_AUC_SCORE"+"\n")
    
    weights = []
    list_index_len = []
    preds_all = []
    preds_all2 = []
    y_all = []
    i = 0
    auc_list = []
    f1_score_list = []
    index_len = 1

    while i < max_len:
        j = 0
        image_test = []
        target_test = []
        while j < len(pd_len):
            val = pd_len.iloc[j]['Len']
            if val == index_len:
                image_test.append(image_all[j])
                target_test.append(y_test[j])
            j = j + 1
        conv_test = np.asarray(image_test)
        conv_test_for_check = np.asarray(image_test)
        conv_test = torch.tensor(conv_test, dtype=torch.float)
        
        pred = model(torch.Tensor(conv_test))
        pred2 = torch.argmax(pred, 1)

        pred = pred.detach().numpy()
        pred2 = pred2.detach().numpy()

        if len(target_test) == 1:
            print("skip")
        else:
            f.write(str(index_len)+";"+str(len(target_test))+";"+str(roc_auc_score(target_test, pred[:, 1]))+"\n")
            auc_list.append(roc_auc_score(target_test, pred[:, 1]))
            f1_score_list.append(f1_score(target_test, pred2))
            list_index_len.append(index_len)
            weights.append(len(target_test))
        index_len = index_len + 1

        i = i + 1


    auc_list = np.asarray(auc_list)
    weights = np.asarray(weights)

    auc_weight = np.sum((auc_list * weights))/np.sum(weights)
    f1_weight = np.sum((f1_score_list * weights))/np.sum(weights)



    print("---METRICS---")
    print("WEIGHTED METRICS")
    print("ROC_AUC_SCORE weighted: %.2f" % auc_weight)
    print("F1_SCORE weighted: %.2f" % f1_weight)

    f.write('ROC_AUC_SCORE weighted;'+str(auc_weight)+";"+"\n")
    f.write('F1_SCORE weighted;'+str(f1_weight)+";"+"\n")
    f.close()

    return list_index_len, auc_list, auc_weight