import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import os
from torch.autograd import Variable
import plotly.graph_objects as go


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def metrics(dataset_name, columns_sel):
    model = torch.load('models/model_' + dataset_name + '.h5')

    pd_len = pd.read_csv("dataset/"+dataset_name+"/len_test"+dataset_name+".csv", header=0, sep=',')
    max_len = pd_len['Len'].max()

    df_test = pd.read_csv("dataset/"+dataset_name+"/"+dataset_name+'_test.csv', header=0, sep=',')
    df_test = df_test[columns_sel]
    
    image_all = df_test.values
    image_all = image_all[:, 0:len(df_test.columns) - 1]

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

    print("---WEIGHTED METRICS---")
    print("ROC_AUC_SCORE weighted: %.2f" % auc_weight)

    f.write('ROC_AUC_SCORE weighted;'+str(auc_weight)+";"+"\n")
    f.write('F1_SCORE weighted;'+str(f1_weight)+";"+"\n")
    f.close()


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list_index_len,
        y=auc_list,
        name = dataset_name,
        connectgaps=True
    ))

    fig.update_layout(
        title=dataset_name,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    #fig.show()