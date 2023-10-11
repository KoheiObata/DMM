import numpy as np
import os
import matplotlib.pyplot as plt


def make_dir(input_dir):
    if os.path.isdir(input_dir):
        print(f'{input_dir} already exist')
    else:
        os.makedirs(f"{input_dir}")
        print(f'{input_dir} is ready')

def get_seq_at_index(X,cp,index,return_index=False):
    #return X and its index(if return_index=True) at segment number(index)
    #X=[n*t,p] n:sample,t:time,p:variable
    #cp:cut point
    if len(cp)==0:
        if return_index:
            return X, np.arange(0,len(X))
        else:
            return X
    else:
        if index==0:
            index_st=0
            index_ed=cp[index]
        elif index>0 and index<len(cp):
            index_st=cp[index-1]
            index_ed=cp[index]
        elif index==len(cp):
            index_st=cp[index-1]
            index_ed=len(X)

        if return_index:
            return X[index_st:index_ed], np.arange(index_st,index_ed)
        else:
            return X[index_st:index_ed]

def get_cov_at_index(Cov,index):
    #return Cov at segment number(index)
    #Cov=[t,p,p]
    return Cov[index,:,:]

def get_cp_len_at_index(index,cp,length):
    if len(cp)==0:
        return length
    else:
        if index==0:
            index_st=0
        else:
            index_st=get_cp_at_index(index-1,cp,length)
        index_ed=get_cp_at_index(index,cp,length)
        return index_ed-index_st

def get_cp_at_index(index,cp,length):
    #return cutpoint at segment number(index)
    if len(cp)==0:
        return 0
    elif index==len(cp):
        return length
    else:
        return cp[index]



#norm X at each cutpoint
def norm_X_at_cp(X,cp):
    norm_X=np.zeros_like(X)
    mean_X=np.zeros_like(X)
    std_X=np.zeros_like(X)
    for i in range(len(cp)+1):
        x,index=get_seq_at_index(X,cp,i,return_index=True)
        mean_x=np.mean(x,axis=0)
        std_x=np.std(x,axis=0)
        mean_X[index]=mean_x
        std_X[index]=std_x
        norm_X[index]=(x-mean_x)/std_x
    np.nan_to_num(norm_X,copy=False,nan=0)
    np.nan_to_num(mean_X,copy=False,nan=0)
    np.nan_to_num(std_X,copy=False,nan=0)
    return norm_X,mean_X,std_X


def data_import(args):
    try:
        X=np.load(args.data_path) #.npy [time,sensor,user]
        X=np.squeeze(X)
    except ValueError:
        X=np.loadtxt(args.data_path) #.txt [time,sensor]

    if args.z_norm:
        X=(X-np.nanmean(X,axis=0))/np.nanstd(X,axis=0)
        np.nan_to_num(X,copy=False,nan=0)

    window=args.window
    cp=np.arange(window,X.shape[0],window)
    if cp[-1]!=window:
        cp=cp[:-1]

    if args.window_z_norm:
        X,mean_d,std_d=norm_X_at_cp(X,cp)

    return X,cp

def save(self,X,history_cp,history):
    ndim = X.ndim - 1

    if X.ndim==3:
        X=X[:,:,0]
    elif X.ndim==4:
        X=X[:,:,0,0]
    elif X.ndim==5:
        X=X[:,:,0,0,0]
    elif X.ndim==6:
        X=X[:,:,0,0,0,0]
    elif X.ndim==7:
        X=X[:,:,0,0,0,0,0]
    elif X.ndim==8:
        X=X[:,:,0,0,0,0,0,0]
    if history:
        num=len(history_cp)
        fig,axes=plt.subplots(nrows=num+1,ncols=1,figsize=(15,4*(num+1)))
        plt.rcParams['font.size']=24
        for i in range(num+1):
            axes[i].plot(X)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].set_xlim(-0.5,len(X)+0.5)
            if i>=1:
                for j in range(len(history_cp[i-1])):
                    axes[i].axvline(x=history_cp[i-1][j], lw=5)
                    axes[i].set_xlabel('Time')
                    axes[i].set_ylabel('Value')
                    axes[i].set_xlim(-0.5,len(X)+0.5)
        fig.tight_layout()
        fig.savefig(f'{self.save_dir}/segmentation.png')
        plt.close()


    make_dir(f'{self.save_dir}/cov')
    transition=np.zeros(len(X))
    for i,regime in enumerate(self.result):
        for n in range(ndim):
            np.savetxt(f'{self.save_dir}/cov/invcov{n}_{i}.txt',regime.Cov[n][0],fmt='%.2f')
            np.savetxt(f'{self.save_dir}/cov/cov{n}_{i}.txt',regime.Cov[n][1],fmt='%.2f')
        transition[regime.index]=i
    np.savetxt(f'{self.save_dir}/transition.txt',transition,fmt='%d')

    fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(10,6))
    axes[0].plot(X)
    axes[1].plot(transition)
    fig.savefig(f'{self.save_dir}/classification.png')
    plt.close()

    np.savetxt(f'{self.save_dir}/costT.txt',np.array(self.min_costT).reshape(1),fmt='%.10e')
    np.savetxt(f'{self.save_dir}/time.txt',np.array(self.time).reshape(1),fmt='%.10e')


def save_gorgeous(args):
    def y_to_cp(y):
        last_y=y[0]
        cp=np.empty(0,dtype=np.int32)
        for i,val_y in enumerate(y):
            if val_y!=last_y:
                cp=np.append(cp,i)
                last_y=val_y
        return cp

    transition=np.loadtxt(f'{args.save_dir}/transition.txt')
    X,cp=data_import(args)
    if X.ndim==3:
        X=X[:,:,0]
    elif X.ndim==4:
        X=X[:,:,0,0]
    elif X.ndim==5:
        X=X[:,:,0,0,0]
    elif X.ndim==6:
        X=X[:,:,0,0,0,0]
    elif X.ndim==7:
        X=X[:,:,0,0,0,0,0]
    elif X.ndim==8:
        X=X[:,:,0,0,0,0,0,0]
    norm_X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    window_X,mean_d,std_d=norm_X_at_cp(X,cp)

    new_cp=y_to_cp(transition)

    color=['y','c','g','r','k','m','b','w']
    color*=int(len(np.unique(transition))/len(color))+1
    plt.rcParams['font.size']=10
    fig,axes=plt.subplots(nrows=6,ncols=1,figsize=(10,12))
    axes[0].plot(X)
    axes[0].set_title('original')
    axes[1].plot(norm_X)
    axes[1].set_title('norm')
    axes[2].plot(mean_d)
    axes[2].set_title('mean')
    axes[3].plot(std_d)
    axes[3].set_title('std')
    axes[4].plot(window_X)
    axes[4].set_title('residue')

    axes[0].axvspan(0,get_cp_at_index(0,new_cp,X.shape[0]),color=color[int(transition[0])],alpha=0.2)
    axes[1].axvspan(0,get_cp_at_index(0,new_cp,norm_X.shape[0]),color=color[int(transition[0])],alpha=0.2)
    axes[4].axvspan(0,get_cp_at_index(0,new_cp,window_X.shape[0]),color=color[int(transition[0])],alpha=0.2)
    axes[5].plot(transition)
    axes[5].set_title('transition')
    for i in range(len(new_cp)):
        st=get_cp_at_index(i,new_cp,X.shape[0])
        ed=get_cp_at_index(i+1,new_cp,X.shape[0])
        axes[0].axvspan(st,ed,color=color[int(transition[st])],alpha=0.2)
        axes[1].axvspan(st,ed,color=color[int(transition[st])],alpha=0.2)
        axes[4].axvspan(st,ed,color=color[int(transition[st])],alpha=0.2)
    fig.legend()
    plt.savefig(f'{args.save_dir}/gorgeous_fig.png')

