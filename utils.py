import pandas as pd
import numpy as np
import os
import re

from sklearn.preprocessing import LabelEncoder



def read_dot_dat_file(path):
    datContent = [i.strip().split() for i in open(path).readlines()]
    r = re.compile("@inputs.*")
    _at_data = datContent.index(['@data'])
    assert datContent[0][0] == '@relation'
    assert datContent[_at_data-1][0] == '@outputs'
    assert datContent[_at_data-2][0] == '@inputs'
    assert len(datContent[_at_data-3][2:]) == 2   # Two Class

    col_names = datContent[_at_data-2][1:]
    col_names.append(datContent[_at_data-1][1])
    
    df = pd.read_csv(path, skiprows=_at_data+1, names=col_names, sep=r', ', engine='python')
    # df = pd.read_csv(path, skiprows=_at_data+1, names=col_names, sep=", ", engine='python')

    class1 = datContent[_at_data-3][2:][0].replace("{","").replace(",","")
    class2 = datContent[_at_data-3][2:][1].replace("}","").replace(",","")

    df['Class'] = df['Class'].replace({class1: 1, class2: -1})
    
    ### convert categorical variables into numerical
    needs_to_convert = []
    for i in range(1,len(col_names)+4):
        if datContent[i][0] == '@attribute' and datContent[i][2] == 'nominal':
            needs_to_convert.append(col_names[i-1])
    
    for need_to_convert in needs_to_convert:
        le = LabelEncoder()
        label = le.fit_transform(df[need_to_convert])        
        df[need_to_convert] = label
    
    return df



def res_to_files(folder, dataset, classifier, dict_res):
    if not os.path.exists('./Results/'+folder+'/'+dataset+'/'+classifier):
        os.makedirs('./Results/'+folder+'/'+dataset+'/'+classifier)
    np.save('./Results/'+folder+'/'+dataset+'/'+classifier+'/mcc.npy',dict_res['mcc'])
    np.save('./Results/'+folder+'/'+dataset+'/'+classifier+'/auc.npy',dict_res['auc'])
    np.save('./Results/'+folder+'/'+dataset+'/'+classifier+'/f1.npy',dict_res['f1'])
    np.save('./Results/'+folder+'/'+dataset+'/'+classifier+'/gmean.npy',dict_res['gmean'])
    np.save('./Results/'+folder+'/'+dataset+'/'+classifier+'/exe_time.npy',dict_res['exe_time'])
    np.save('./Results/'+folder+'/'+dataset+'/'+classifier+'/y_pred.npy',dict_res['y_pred'])
    
    
