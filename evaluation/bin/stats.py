from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import csv
import ast

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.preprocessing\._label"
)


def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
        
    return label_pos_dict

def get_eval_metrics(pred_label, pred_probs, true_label, all_label):
    mlb = MultiLabelBinarizer()

   
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    label_pos_dict = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
        labels, probs = pred_label[i], pred_probs[i]
        for label, prob in zip(labels, probs):
            if label in all_label:
                pos = label_pos_dict[label]
                pred_m_auc[i, pos] = prob
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted', zero_division=0)
    f1 = f1_score(true_m, pred_m, average='weighted')
    try:
        roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    except:
        roc = 0
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, roc, acc, true_m, pred_m, pred_m_auc

def get_true_labels_orig(file_name):
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label
def get_pred_labels(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label

def get_pred_probs(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_probs = []
    for row in csvreader:
        preds_with_dist = row[2:]
        probs = np.zeros(len(preds_with_dist))
        count = 0
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = float(pred_ec_dist.split(":")[1].split("/")[1])
            probs[count] = ec_i
            #preds_ec_lst.append(probs)
            count += 1
        # sigmoid of the negative distances 
        probs = (1 - np.exp(-1/probs)) / (1 + np.exp(-1/probs))
        probs = probs/np.sum(probs)
        pred_probs.append(probs)
    return pred_probs


def get_true_labels(file_name):
    datafile = pd.read_csv(file_name)
    datafile["ActSeekN Prediction"] = datafile["ActSeekN Prediction"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )    
    all_label = set()
    true_label_dict = {}
    predictions=[]
    pred_probs=[]
    count=0
    for k, row in datafile.iterrows():        
        pred=[]
        probs=[] 
        try:
            for f in row["ActSeekN Prediction"]:
                ec = f[0].split(" -- ")[1].split("-")[0]                   
                        
                if '|' in ec:   
                    for e in ec.split("|"):
                        if len(e) > 1 and e not in pred:
                            ecchecking = e.split(".")
                            ecchecking = ecchecking[len(ecchecking)-1]
                            if len(pred)>5:
                                continue
                            if f[1]> 0.15:
                                pred.append(e)
                                probs.append(f[1]) 
                            
                else:
                    if ec not in pred:
                        if f[1]> 0.15:
                            pred.append(ec)
                            probs.append(f[1])   
                            count=count+1
                
        except:
            pred.append("")
            probs.append(1)
        
        if len(row["EC Numbers"])>0:
            predictions.append(pred)
            pred_probs.append(probs)
            true_ec_lst = row["EC Numbers"].split(';')
            true_label_dict[row["Entry"]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
       
   
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    
    return true_label, all_label, predictions, pred_probs




def get_true_labels_clean(file_name):
    datafile = pd.read_csv(file_name)
    all_label = set()
    true_label_dict = {}    
    predictions=[]
    pred_probs=[]
    for k, row in datafile.iterrows():
        pred=[]
        probs=[]
        for f in row["CLEAN-contact Prediction"].split(','):   
            ec = f.replace("EC:","")                     
            if ec not in pred:
                pred.append(ec)
                probs.append(1) 
            else:
                pred.append("")
                probs.append(1)
                
        if len(row["EC Numbers"])>0:                
            predictions.append(pred)
            pred_probs.append(probs)
            true_ec_lst = row["EC Numbers"].split(';')
            true_label_dict[row["Entry"]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)  
      
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label, predictions, pred_probs