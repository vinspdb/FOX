import streamlit as st
import numpy as np
import pandas as pd
import load_weights
import torch
from PIL import Image
st.set_page_config(page_title='FOX')
image = Image.open('fox.png')
st.image(image)

dataset_name = st.sidebar.selectbox('Choose Dataset',
    ('sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4',
     'bpic2011_f1', 'bpic2011_f2', 'bpic2011_f3', 'bpic2011_f4',
     'bpic2012_accepted', 'bpic2012_cancelled', 'bpic2012_declined','production')
    )

st.title('Performace on '+str(dataset_name) + ' event log')

if dataset_name == 'sepsis_cases_1':
        columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'std_Leucocytes', 'std_CRP', 'Classification']#sepsis_cases_1
elif dataset_name == 'sepsis_cases_2':
        columns_sel = ['Diagnose', 'mean_open_cases', 'mean_hour', 'DisfuncOrg', 'Classification']#sepsis_cases_2
elif dataset_name == 'sepsis_cases_4':
        columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'org:group_E', 'std_CRP', 'DiagnosticECG', 'Classification']#sepsis_cases_4
elif dataset_name == 'bpic2011_f1':
        columns_sel = ['Diagnosis Treatment Combination ID', 'mean_open_cases', 'Diagnosis', 'Activity code_376400.0', 'Classification']#bpic2011_f1
elif dataset_name == 'bpic2011_f2':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'Diagnosis code', 'mean_open_cases', 'Activity code_376400.0', 'Age', 'Producer code_CHE1', 'Classification']#bpic2011_f2
elif dataset_name == 'bpic2011_f3':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'mean_open_cases', 'Diagnosis code', 'std_event_nr', 'mean_event_nr', 'Classification']#bpic2011_f3
elif dataset_name == 'bpic2011_f4':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Treatment code', 'Classification']#bpic2011_f4
elif dataset_name == 'bpic2012_accepted':
        columns_sel = ['AMOUNT_REQ', 'Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE', 'Activity_W_Valideren aanvraag-START','Classification']#bpic2012_accepted
elif dataset_name == 'bpic2012_declined':
        columns_sel = ['AMOUNT_REQ', 'Activity_A_PARTLYSUBMITTED-COMPLETE', 'Activity_A_PREACCEPTED-COMPLETE', 'Activity_A_DECLINED-COMPLETE', 'Activity_W_Completeren aanvraag-SCHEDULE', 'mean_open_cases', 'Classification'] #bpic2012_declined
elif dataset_name == 'bpic2012_cancelled':
        columns_sel = ['Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE', 'Activity_W_Valideren aanvraag-START', 'AMOUNT_REQ', 'Activity_W_Valideren aanvraag-COMPLETE', 'Activity_A_CANCELLED-COMPLETE', 'Classification']#bpic2012_cancelled
elif dataset_name == 'production':
        columns_sel = ['Work_Order_Qty', 'Activity_Turning & Milling - Machine 4', 'Resource_ID0998', 'Resource_ID4794', 'Resource.1_Machine 4 - Turning & Milling', 'Classification']#production

df_test = pd.read_csv("dataset/"+dataset_name+"/"+dataset_name+'_test.csv', header=0, sep=',')


with st.form('performace'):

    submit_perf = st.form_submit_button('Get results')

if submit_perf:
        list_index_len, auc_list, auc_weight = load_weights.metrics(dataset_name, columns_sel)
        chart_data = pd.DataFrame(
            auc_list,
            columns=['AUC'])
        st.header('AUC: ' + str(round(auc_weight, 2)))

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list_index_len,
            y=auc_list,
            name=dataset_name,
            connectgaps=True
        ))

        st.plotly_chart(fig)



st.title('Trace outcome prediction')
st.header('Please fill the boxes')

features_dictionary = dict.fromkeys(columns_sel[:len(columns_sel)-1], "")

with st.form('Form1'):
    for k, v in features_dictionary.items():
        features_dictionary[k] = st.text_input(k, v)
        st.write(features_dictionary[k])
    submit = st.form_submit_button('Predict')

if submit:
        i = 0
        list_val = []
        while i < len(columns_sel[:len(columns_sel)-1]):
            list_val.append(float(features_dictionary[columns_sel[i]]))
            i = i + 1
        print(list_val)
        list_val = np.array(list_val)
        model = torch.load('models/model_'+dataset_name + '.h5')
        pred = model(torch.Tensor([list_val]))
        pred2 = torch.argmax(pred, 1)

        pred2 = pred2.detach().numpy()

        if pred2 == 0:
            res = 'regular'
        else:
            res = 'deviant'

        st.text('This trace is '+ str(res))
        rule, firerule, index_rule = load_weights.get_fire_strength(model,pred2)
        st.write('Because:')  # df, err, func, keras!

        list_max_index =[]
        list_max_fire = []
        for r in index_rule:
            list_max_index.append('RULE '+ str(r))
            list_max_fire.append(firerule.get(r))

        rule = rule.replace('mf0','low')
        rule = rule.replace('mf1','medium')
        rule = rule.replace('mf2','high')

        i = 0
        while i < len(columns_sel[:len(columns_sel)-1]):
            rule = rule.replace('x'+str(i), columns_sel[i])
            i = i + 1
        c = rule.split('and')
        i = 0
        list_exp = []
        while i<len(c):
            if i == 0:
                c[i] = c[i][c[i].find('IF'):]
                c[i] = c[i].replace('IF','')
            if i == len(c)-1:
                c[i] = c[i].split('THEN')
                c[i] = c[i][0]

            list_exp.append(c[i])

            print(c[i])
            i = i + 1

        st.write(list_exp)

        import plotly.graph_objects as go

        barplot = go.Figure(go.Bar(
            x=list_max_fire[:5],
            y=list_max_index[:5],
            orientation='h',
            ),
            layout=go.Layout(
                title="Top 5 rules",
                xaxis=dict(
                    title="Fire strenght"
                ),
                yaxis=dict(
                    title="Rules"
                ))
        )

        st.plotly_chart(barplot)

