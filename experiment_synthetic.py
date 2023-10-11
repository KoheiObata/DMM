import numpy as np
import os

from src import model



def make_dir(input_dir):
    if os.path.isdir(input_dir):
        print(f'{input_dir} already exist')
    else:
        os.makedirs(f"{input_dir}")
        print(f'{input_dir} is ready')


def experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_name,w_list,sparsity_list):
    for A_size in A_size_list:
        for B_size in B_size_list:
            for seg_length in seg_length_list:
                for name in pattern_name:
                    for i in range(count):
                        save_dir=f'{SAVE_DIR}/d1={A_size}/d2={B_size}/t={seg_length}/{name}/{i}'
                        execute(save_dir,w_list,sparsity_list)


def experiment_matrix(SAVE_DIR,count,matrix_size_list,seg_length_list,pattern_name,w_list,sparsity_list):
    for matrix_size in matrix_size_list:
        for seg_length in seg_length_list:
            for name in pattern_name:
                for i in range(count):
                    save_dir=f'{SAVE_DIR}/n={len(matrix_size)}/d={matrix_size[0]}/t={seg_length}/{name}/{i}'
                    execute(save_dir,w_list,sparsity_list)

def execute(save_dir,w_list,sparsity_list):
    data_path=f'{save_dir}/data.npy'
    label_path=f'{save_dir}/label.txt'

    datadir=save_dir.replace('./data','')
    for w in w_list:
        for sparsity in sparsity_list:
            param_dir=f'w={w}/sparsity={sparsity}'
            data_name=os.path.join(datadir, param_dir)
            cwd = os.getcwd()
            aaa=f'{cwd}/result/{data_name}/'
            if os.path.isfile(f'{aaa}/0/f1.txt'):
                print('this experiment is already done')
            else:
                print('start experiment')
                model.main(data_path,data_name,sparsity,window=w,label_path=label_path,cf=32,z_norm=False,window_z_norm=False,evaluate=True)



def experiment(experiment_id):
    basedir='./data'

    if experiment_id==0:
        SAVE_DIR=f'{basedir}/datatype2D'
        A_size_list=[10]
        B_size_list=[1]
        seg_length_list=[100]
        pattern_name=['patternA','patternB','patternC','patternD']
        count=10
        w_list=[4]
        sparsity_list=[4]
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_name,w_list,sparsity_list)

    elif experiment_id==1:
        SAVE_DIR=f'{basedir}/datatype3D'
        A_size_list=[10]
        B_size_list=[10]
        seg_length_list=[100]
        pattern_name=['patternA','patternB','patternC','patternD']
        count=10
        w_list=[4]
        sparsity_list=[4]
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_name,w_list,sparsity_list)

    elif experiment_id==2:
        SAVE_DIR=f'{basedir}/varyd1'
        A_size_list=list(np.arange(5,51,5))
        B_size_list=[5]
        seg_length_list=[100]
        pattern_name=['patternC']
        count=10
        w_list=[4]
        sparsity_list=[4]
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_name,w_list,sparsity_list)

    elif experiment_id==3:
        SAVE_DIR=f'{basedir}/varyt'
        A_size_list=[5]
        B_size_list=[5]
        seg_length_list=[100,250,500,750,1000,2500,5000,7500,10000]
        pattern_name=['patternC']
        count=1
        w_list=[4]
        sparsity_list=[4]
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_name,w_list,sparsity_list)

    elif experiment_id==4:
        SAVE_DIR=f'{basedir}/2Dvaryd1'
        A_size_list=list(np.arange(5,51,5))
        B_size_list=[1]
        seg_length_list=[100]
        pattern_name=['patternC']
        count=10
        w_list=[4]
        sparsity_list=[4]
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_name,w_list,sparsity_list)

id=0
experiment(id)
# for id in [0,1,2,3,4]:
    # experiment(id)