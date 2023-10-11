import numpy as np

from sklearn.metrics import confusion_matrix ,accuracy_score, f1_score
import itertools

def confusion_matrix_to_truepred(con_matrix):
    y_true=np.empty(0)
    y_pred=np.empty(0)
    for true in range(con_matrix.shape[0]):
        for pred in range(con_matrix.shape[1]):
            y_true=np.concatenate([y_true,np.ones(con_matrix[true,pred])*true])
            y_pred=np.concatenate([y_pred,np.ones(con_matrix[true,pred])*pred])
    return y_true, y_pred

def evaluate_clustering_accuracy(y_true,y_pred):
    true_cluster_num=len(np.unique(y_true))
    pred_cluster_num=len(np.unique(y_pred))
    if pred_cluster_num>20:
        max_acc, max_f1, max_cm=0,0,0
    elif true_cluster_num<pred_cluster_num:
        max_acc, max_f1, max_cm=evaluate_clustering_accuracy_fast(y_true,y_pred)
    else:
        max_acc, max_f1, max_cm=evaluate_clustering_accuracy_few(y_true,y_pred)
    return max_acc, max_f1, max_cm

def evaluate_clustering_accuracy_few(y_true,y_pred):
    true_cluster_num=len(np.unique(y_true))
    pred_cluster_num=len(np.unique(y_pred))

    cm=confusion_matrix(y_true,y_pred)
    seq=set(range(cm.shape[0]))
    max_f1,max_acc,max_cm=0,0,0
    for s in list(itertools.permutations(seq)):
        cm_temp=cm[s,:]
        y_true_temp,y_pred_temp=confusion_matrix_to_truepred(cm_temp)
        acc_temp=accuracy_score(y_true_temp,y_pred_temp)
        f1_temp=f1_score(y_true_temp,y_pred_temp,average='macro')
        #when 'macro', a label that has no sample is also counted as an effective label.
        if true_cluster_num<pred_cluster_num:
            f1_temp=f1_temp*pred_cluster_num/true_cluster_num
        if  f1_temp>max_f1:
            max_acc=acc_temp
            max_f1=f1_temp
            max_cm=cm_temp
    return max_acc, max_f1, max_cm

def evaluate_clustering_accuracy_fast(y_true,y_pred):
    true_cluster_num=len(np.unique(y_true))
    pred_cluster_num=len(np.unique(y_pred))

    cm=confusion_matrix(y_true,y_pred)
    m=true_cluster_num+2
    important_pred=np.argsort(np.mean(cm,axis=0))[::-1][:m]


    n=3

    true_pred=[]
    tentative=-1
    pos_n=1
    for pred in range(pred_cluster_num):
        if pred in important_pred:
            top_n=np.argsort(cm[:,pred])[::-1]
            for true in top_n[:n]:
                if cm[pred,true]>0:
                    pos_n+=1
            if len(top_n[:pos_n])==0:
                true_pred.append([tentative])
                tentative-=1
            else:
                true_pred.append(top_n[:pos_n])
        else:
            true_pred.append([tentative])
            tentative-=1
        pos_n=1

    max_f1,max_acc,max_cm=0,0,0
    cm_diag=0
    for seq_seed in list(itertools.product(*true_pred)):
        seq_seed=np.array(seq_seed)
        random_pick=set(range(pred_cluster_num))-set(np.unique(seq_seed))
        u, counts=np.unique(seq_seed,return_counts=True)

        true_pred_inner=[[] for _ in range(pred_cluster_num)]
        for i,uni in enumerate(u[counts!=1]):
            rand=list(random_pick)[:int(counts[np.where(u==uni)[0]])-1]
            random_pick=set(random_pick)-set(rand)
            rand=list(set(rand) | set([uni]))
            for j,seed in enumerate(seq_seed):
                if seed==uni:
                    true_pred_inner[j]=rand


        for j,seed in enumerate(seq_seed):
            if seed in u[counts!=1]:
                pass
            elif seed<0:
                rand=list(random_pick)[:1]
                random_pick=set(random_pick)-set(rand)
                true_pred_inner[j]=rand
            else:
                true_pred_inner[j]=[seed]
        for s in list(itertools.product(*true_pred_inner)):
            if len(np.unique(s))!=pred_cluster_num:
                continue
            cm_temp=cm[s,:]
            if np.diag(cm_temp).sum()<cm_diag:
                pass
            else:
                y_true_temp,y_pred_temp=confusion_matrix_to_truepred(cm_temp)
                acc_temp=accuracy_score(y_true_temp,y_pred_temp)
                f1_temp=f1_score(y_true_temp,y_pred_temp,average='macro')
                #when 'macro', a label that has no sample is also counted as an effective label.
                if true_cluster_num<pred_cluster_num:
                    f1_temp=f1_temp*pred_cluster_num/true_cluster_num
                if  f1_temp>max_f1:
                    max_acc=acc_temp
                    max_f1=f1_temp
                    max_cm=cm_temp
                    cm_diag=np.diag(max_cm).sum()
    return max_acc,max_f1,max_cm

