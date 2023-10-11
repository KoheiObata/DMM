import numpy as np
import os

def genInvCov(size, low = 0.3 , upper = 0.6, portion = 0.2,symmetric = True):
    portion = portion/2
    A=np.zeros([size,size])
    if size==1:
        return A
    for i in range(size):
        for j in range(size):
            if i>=j:
                continue
            coin=np.random.uniform()
            if coin<portion:
                value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
                A[i,j] = value
    if np.allclose(A,np.zeros([size,size])):
        i,j=0,0
        while i==j:
            i=np.random.randint(0,size,1)
            j=np.random.randint(0,size,1)
        value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
        A[i,j] = value
    if symmetric:
        A = A + A.T
    return np.matrix(A)

def AB_to_C(A_matrix,B_matrix):
    A_size=A_matrix.shape[0]
    B_size=B_matrix.shape[0]
    C_matrix=np.zeros([A_size*B_size,A_size*B_size])
    for i in range(B_size):
        C_matrix[A_size*i:A_size*(i+1),A_size*i:A_size*(i+1)]=A_matrix

    for i in range(B_size):
        for j in range(B_size):
            if i==j:
                continue
            B_ij=B_matrix[i,j]
            B_eye=np.identity(A_size)*B_ij
            C_matrix[A_size*i:A_size*(i+1),A_size*j:A_size*(j+1)]=B_eye
    return C_matrix

def C_to_AB(A_size,B_size,C_matrix):
    A_matrix=C_matrix[:A_size,:A_size]
    B_matrix=np.zeros([B_size,B_size])
    for k in range(B_size):
        for l in range(B_size):
            B_matrix[k,l]=np.mean(np.diag(C_matrix[A_size*k:A_size*(k+1),A_size*l:A_size*(l+1)]))
    return A_matrix, B_matrix

def convert_add_lmd(mat, eps=0.1):
    """
    K' = K + \lmd I
    \lmd is decided as all eigen values are larger than 0
    """
    mat_positive_definite = np.copy(mat)
    eigen_values = np.linalg.eigvals(mat)
    min_eigen_values = np.min(eigen_values)
    if True:
        lmd = np.abs(min_eigen_values) + eps  # new eigen values are larger than 0
        # print('min_eigen_values',min_eigen_values)
        np.add(mat_positive_definite,lmd.real * np.eye(mat.shape[0]),out=mat_positive_definite,casting='unsafe')
    return mat_positive_definite

def genarateData(matrix_size,size_list,pattern,rand_seed=None):
    np.random.seed(rand_seed)
    pattern_type=np.unique(np.array(pattern))
    C_cov_list=[[] for _ in range(len(pattern_type))]
    for pa in np.unique(np.array(pattern)):
        A_inv=genInvCov(matrix_size[0], low = 0.3 , upper = 0.6, portion = 0.2,symmetric = True)
        for size in matrix_size[1:]:
            B_inv=genInvCov(size, low = 0.3 , upper = 0.6, portion = 0.2,symmetric = True)
            A_inv=AB_to_C(A_inv,B_inv)
        C_inv=convert_add_lmd(A_inv)
        C_cov=np.linalg.inv(C_inv)
        C_cov_list[pa]=C_cov

    Data=np.empty([0,np.prod(matrix_size)])
    for pa,size in zip(pattern,size_list):
        cov=C_cov_list[pa]
        mean = np.zeros(np.prod(matrix_size))
        ##Generate data
        data=np.random.multivariate_normal(mean,cov,size)
        Data=np.concatenate([Data,data],axis=0)
    return Data, C_cov_list

def gen_Template(save_dir,matrix_size,size_list,pattern,rand_seed):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data,C_cov_list=genarateData(matrix_size,size_list=size_list,pattern=pattern,rand_seed=rand_seed)

    for i, ms in enumerate(reversed(matrix_size[1:]), 1):
        data = np.stack(np.split(data,ms,axis=-i), axis=-i)
    np.save(f'{save_dir}/data.npy',data)

    y_true=np.zeros(0)
    for p,seg in enumerate(pattern):
        y_true=np.append(y_true, np.ones(size_list[p])*(seg))
    np.savetxt(f'{save_dir}/label.txt',y_true,fmt="%d")

    for j in range(len(C_cov_list)):
        np.savetxt(f'{save_dir}/covC_{j}.txt',np.round(C_cov_list[j],decimals=5),fmt="%.5e")
        np.savetxt(f'{save_dir}/invcovC_{j}.txt',np.round(np.linalg.inv(C_cov_list[j]),decimals=5),fmt="%.5e")

        invcovC=np.linalg.inv(C_cov_list[j])
        for i,ms in enumerate(reversed(matrix_size[1:]),1):
            invcovC, invcovB = C_to_AB(np.prod(matrix_size[:-i]), ms, invcovC)
            np.savetxt(f'{save_dir}/invcov{len(matrix_size)-(i-1)}_{j}.txt',np.round(invcovB,decimals=5),fmt="%.5e")
        np.savetxt(f'{save_dir}/invcov{1}_{j}.txt',np.round(invcovC,decimals=5),fmt="%.5e")

def get_bp(pattern, seg_length):
    id_seg_list, num_seg_list = np.unique(pattern,return_counts=True)
    bp=np.zeros(len(pattern),dtype=int)
    seg_len=np.zeros(len(pattern),dtype=int)
    seg_length_list=[]
    for id_seg, num_seg in zip(id_seg_list,num_seg_list):
        total_seg_length=seg_length*num_seg
        tmp_a=np.random.choice(np.arange(1,total_seg_length),size=num_seg-1,replace=False)
        tmp_a=np.append(tmp_a,total_seg_length)
        tmp_a=np.append(0,tmp_a)
        tmp_a=np.sort(tmp_a)[::-1]
        tmp_b=[]
        for i in range(len(tmp_a)-1):
            tmp_b.append(tmp_a[i]-tmp_a[i+1])
        seg_length_list.append(tmp_b)

    for k,p in enumerate(pattern):
        r=seg_length_list[p].pop()
        bp[k:]+=r
        seg_len[k]=r
    return bp, seg_len

def experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_list,pattern_name):
    rand_seed=100
    for A_size in A_size_list:
        for B_size in B_size_list:
            for seg_length in seg_length_list:
                for pattern,name in zip(pattern_list,pattern_name):
                    for i in range(count):
                        bp, size_list = get_bp(pattern, seg_length)

                        save_dir=f'{SAVE_DIR}/d1={A_size}/d2={B_size}/t={seg_length}/{name}/{i}'
                        gen_Template(save_dir,[A_size,B_size],size_list,pattern,rand_seed)
                        print(sum(size_list),pattern,rand_seed)
                        rand_seed+=1

def experiment_matrix(SAVE_DIR,count,matrix_size_list,seg_length_list,pattern_list,pattern_name):
    rand_seed=100
    for matrix_size in matrix_size_list:
        for seg_length in seg_length_list:
            for pattern,name in zip(pattern_list,pattern_name):
                for i in range(count):
                    bp, size_list = get_bp(pattern, seg_length)

                    save_dir=f'{SAVE_DIR}/n={len(matrix_size)}/d={matrix_size[0]}/t={seg_length}/{name}/{i}'
                    gen_Template(save_dir,matrix_size,size_list,pattern,rand_seed)
                    print(sum(size_list),pattern,rand_seed)
                    rand_seed+=1




def generate_data(experiment_id):
    count=10
    patternA=[0,1,0]
    patternB=[0,1,2,1,0]
    patternC=[0,1,2,3,0,1,2,3]
    patternD=[0,1,1,0,2,2,2,0]

    basedir='.'
    if experiment_id==0:
        SAVE_DIR=f'{basedir}/datatype2D'
        A_size_list=[10]
        B_size_list=[1]
        seg_length_list=[100]
        pattern_list=[patternA,patternB,patternC,patternD]
        pattern_name=['patternA','patternB','patternC','patternD']
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_list,pattern_name)

    elif experiment_id==1:
        SAVE_DIR=f'{basedir}/datatype3D'
        A_size_list=[10]
        B_size_list=[10]
        seg_length_list=[100]
        pattern_list=[patternA,patternB,patternC,patternD]
        pattern_name=['patternA','patternB','patternC','patternD']
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_list,pattern_name)

    elif experiment_id==2:
        SAVE_DIR=f'{basedir}/varyd1'
        A_size_list=list(np.arange(5,51,5))
        B_size_list=[5]
        seg_length_list=[100]
        pattern_list=[patternC]
        pattern_name=['patternC']
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_list,pattern_name)

    elif experiment_id==3:
        SAVE_DIR=f'{basedir}/varyt'
        A_size_list=[5]
        B_size_list=[5]
        seg_length_list=[100,250,500,750,1000,2500,5000,7500,10000]
        pattern_list=[patternC]
        pattern_name=['patternC']
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_list,pattern_name)

    elif experiment_id==4:
        SAVE_DIR=f'{basedir}/2Dvaryd1'
        A_size_list=list(np.arange(5,51,5))
        B_size_list=[1]
        seg_length_list=[100]
        pattern_list=[patternC]
        pattern_name=['patternC']
        experiment_AB(SAVE_DIR,count,A_size_list,B_size_list,seg_length_list,pattern_list,pattern_name)




id=0
generate_data(id)
# for id in [0,1,2,3,4]:
    # generate_data(id)