import numpy as np
import time
import os
import glob

try:
    from tvgl import TVGL as TimeGraphicalLasso
    import util
    import evaluation
except:
    from src.tvgl import TVGL as TimeGraphicalLasso
    from src import util
    from src import evaluation


#CutPoint
class CutPoint():
    '''
    cp: cut point, [5, 10, 15, ...]
    y: segment index [1,1,1,1,2,2,2,2, ....]
    seg_len: segment length [5, 5, 5, ...]
    length: length of sequence
    '''
    def __init__(self,cp=None,y=None,seg_len=None,length=None):
        if isinstance(cp,np.ndarray) and length!=None:
            self.cp=cp
            self.length=length
            self.y=self.cp_to_y(cp=self.cp,length=self.length)
            self.seg_len=self.cp_to_seg(cp=self.cp,length=self.length)
        elif isinstance(y,np.ndarray):
            self.y=y
            self.length=len(y)
            self.cp=self.y_to_cp(y=self.y)
            self.seg_len=self.cp_to_seg(cp=self.cp,length=self.length)
        elif isinstance(seg_len,np.ndarray):
            self.seg_len=seg_len
            self.length=np.sum(seg_len)
            self.cp=self.seg_to_cp(seg_len=self.seg_len)
            self.y=self.cp_to_y(cp=self.cp,length=self.length)
        self.num_seg=len(self.cp)+1

    @classmethod
    def y_to_cp(self,y):
        last_y=y[0]
        cp=np.empty(0,dtype=np.int32)
        for i,val_y in enumerate(y):
            if val_y!=last_y:
                cp=np.append(cp,i)
                last_y=val_y
        return cp

    @classmethod
    def cp_to_y(self,cp,length):
        #length: length of X
        y=np.zeros(length)
        last_c=0
        for i,c in enumerate(cp):
            y[last_c:c]=i
            last_c=c
        y[last_c:]=i+1
        return y

    @classmethod
    def cp_to_seg(self,cp,length):
        seg_len=np.empty(0,dtype=np.int32)
        seg_len=np.append(seg_len,cp[0])
        for i in range(1,len(cp)):
            seg_len=np.append(seg_len,cp[i]-cp[i-1])
        seg_len=np.append(seg_len,length-cp[len(cp)-1])
        return seg_len

    @classmethod
    def seg_to_cp(self,seg_len):
        cp=np.empty(0,dtype=np.int32)
        for i in range(1,len(seg_len)):
            cp=np.append(cp,np.sum(seg_len[:i]))
        return cp

    def cp_value(self,index):
        return util.get_cp_at_index(index,self.cp,self.length)

    def even_cp(self):
        return CutPoint(cp=self.cp[1::2],length=self.length)

    def odd_cp(self):
        return CutPoint(cp=self.cp[::2],length=self.length)


#GraphicalLasso
class Segment():
    def __init__(self,x,gl_mode=None,index=None,id=None,Cov=None,sparsity=0.1,max_iter=100):
        '''
        input
        x: input tensor [time, d1, d2, ..., dn]
        gl_mode: [0, 0, 1, ..., 0] if 1 use different gl inference
        index: index of data [1,2,3, ...]
        Cov: list of covariance matrix and presicion matrix [([d1, d1], [d1, d1]), ..., ([dn, dn], [dn, dn])]
        '''

        self.index = index
        self.index_list = consecutive_index(index)
        self.id = id
        self.sparsity = sparsity
        self.max_iter = max_iter
        if isinstance(Cov,list):
            self.Cov = Cov
        else:
            self.Cov = multimode_GL(x,gl_mode,sparsity=self.sparsity,max_iter=self.max_iter)
        self.costT,self.costM,self.costC,self.costA = segment_MDL(x,gl_mode,self.index_list,self.Cov,self.sparsity)

def consecutive_index(index):
    '''find groups of consecutive indices in a given list index

    for example
    input [1,2,3,5,6,8,9]
    output [[1,2,3], [5,6], [8,9]]
    '''
    result = []
    tmp = [index[0]]
    for i in range(len(index)-1):
        if index[i+1] - index[i] == 1:
            tmp.append(index[i+1])
        else:
            if len(tmp) > 0:
                result.append(tmp)
            tmp = []
            tmp.append(index[i+1])
    result.append(tmp)
    return result

def Segments_GL(X,cp,gl_mode,sparsity,max_iter):
    Segments=[]
    for i in range(len(cp)+1):
        X_seg,ind=util.get_seq_at_index(X,cp,index=i,return_index=True)
        Segments.append(Segment(X_seg,gl_mode,index=ind,id=i,sparsity=sparsity,max_iter=max_iter))
    return Segments

def multimode_GL(X, gl_mode, sparsity=0.1, max_iter=100):
    '''
    input
    X: input tensor [time, d1, d2, ..., dn]
    sparsity: TVGL parameter
    max_iter: TVGL parameter

    TVGL parameter
    alpha: sparsity of precision matrix
    beta: temporal similarity
    psi: penalty function
    max_iter: maximum iteration time until convergence
    '''
    ndim = X.ndim - 1

    Cov = []
    for n in range(ndim):
        X_unfold = sf_unfold(X, gl_mode, mode=n)
        TVGL=TimeGraphicalLasso(alpha=sparsity,beta=0,max_iter=max_iter,psi='laplacian',assume_centered=False)
        TVGL.fit(X_unfold,np.zeros(X_unfold.shape[0]))
        cov=(TVGL.precision_[0],TVGL.covariance_[0]) # [dn, dn]
        Cov.append(cov)
    return Cov

def f_unfold(tensor, mode=0, beginning=True):
    """Unfolds a tensors following the Kolda and Bader definition

        Moves the `mode` axis to the beginning (if beginning) and reshapes in Fortran order
    """
    if beginning:
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')
    else:
        return np.reshape(np.moveaxis(tensor, mode, -1), (-1, tensor.shape[mode]), order='F')

def sf_unfold(tensor, gl_mode=None, mode=0):
    '''
    for example
    input
        tensor.shape = [10, 2,3,4,5]
        gl_mode = [0,1,1,0]
        mode = 0
    return
        X.shape = [10*5, 3*4, 2]

    input
        tensor.shape = [10, 2,3,4,5]
        gl_mode = [0,1,1,0]
        mode = 2
    return
        X.shape = [10*2*5, 3, 4]
    '''
    ndim=tensor.ndim -1
    if not gl_mode:
        gl_mode = [1 for i in range(ndim)]

    indices = np.where(np.array(gl_mode) == 1)
    indices = list(indices[0])
    if gl_mode[mode] == 0:
        rest = list(set(np.arange(ndim)) - set([*indices,mode]))
        rest_indices = indices
        order = [*rest,*rest_indices,mode]
    elif gl_mode[mode] == 1:
        rest = list(set(np.arange(ndim)) - set([*indices]))
        rest_indices = list(set([*indices]) - set([mode]))

    order = [*rest,*rest_indices,mode]
    order = [val+1 for val in order]
    X = np.transpose(tensor, [0,*order])
    id_dim = 1
    for id in rest_indices:
        id_dim *= tensor.shape[id+1]
    X = X.reshape(-1, id_dim, tensor.shape[mode+1])
    return np.squeeze(X)

#MDL
def segments_MDL_dam(Segments,X,gl_mode):
    # Segments[solo,solo,solo]
    # return costT of each segment
    costT=0
    costA,costC,costM,costL=0,0,0,0
    n_cluster=len(Segments)
    n_segment=0
    for seg in Segments:
        T,M,C,A,L=segment_MDL_dam(X[seg.index],gl_mode,seg.index_list,seg.Cov,seg.sparsity)
        n_segment+=len(seg.index_list)
        costT+=T
        costM+=M
        costA+=A
        costC+=C
        costL+=L
    costT+=log_s(n_segment)+log_s(n_cluster)+n_segment*log_s(n_cluster) # number of segments and cluster + assignment
    costA+=log_s(n_segment)+log_s(n_cluster)+n_segment*log_s(n_cluster) # number of segments and cluster + assignment
    return costT,costM,costC,costA,costL
def segment_MDL_dam(x,gl_mode,index_list,Cov,sparsity):
    ndim = x.ndim - 1
    costM,costC,costL = 0, 0, 0
    costA = data_length_cost(index_list)
    for n in range(ndim):
        x_unfold = sf_unfold(x, gl_mode, mode=n) # [sample, Dn, dn]
        costM += model_description_cost(Cov[n][0])
        c_temp,l_temp = data_coding_cost_dam(x_unfold,Cov[n][0],sparsity)
        costC+=c_temp
        costL+=l_temp
    costM/=ndim
    costT = costA+costM+costC+costL
    return costT,costM,costC,costA,costL
def data_coding_cost_dam(x,cov,sparsity):
    return -likelihood(x,cov), sparsity*l1_norm(cov)

def segments_MDL(Segments,X,gl_mode):
    # Segments[solo,solo,solo]
    # return costT of each segment
    costT=0
    n_cluster=len(Segments)
    n_segment=0
    for seg in Segments:
        T,_,_,_=segment_MDL(X[seg.index],gl_mode,seg.index_list,seg.Cov,seg.sparsity)
        n_segment+=len(seg.index_list)
        costT+=T
    costT+=log_s(n_segment)+log_s(n_cluster)+n_segment*log_s(n_cluster) # number of segments and cluster + assignment

    return costT

def segment_MDL(x,gl_mode,index_list,Cov,sparsity):
    '''
    input
    x: input tensor (concatenated segment)
    index_list: length of each segment
    Cov: covariance of each mode
    '''

    ndim = x.ndim - 1
    costM,costC = 0, 0
    costA = data_length_cost(index_list)
    for n in range(ndim):
        x_unfold = sf_unfold(x, gl_mode, mode=n) # [sample, Dn, dn]
        costM += model_description_cost(Cov[n][0])
        costC += data_coding_cost(x_unfold,Cov[n][0],sparsity)
    costM/=ndim
    costT = costA+costM+costC
    return costT,costM,costC,costA

def model_description_cost(cov):
    p=cov.shape[0]
    n_p=p*(p-1)/2 # size of upper triangular elements (without diagonal component)
    cov=np.triu(cov,k=1) # upper triangular (without diagonal component)

    non_zero=np.count_nonzero(abs(cov)>threshold) # number of nonzero elements
    if non_zero<=0 or cov.shape==(1,1):
        return 0
    else:
        cov_cost=non_zero*(np.log2(n_p)+CF)+log_s(non_zero) # cost of location of each value
        cov_cost+=p*(np.log2(p)+CF) # cost of storing diagonal value
        cov_cost/=p**2 # normalization with the dimension size
    return cov_cost

def data_coding_cost(x,cov,sparsity):
    return -likelihood(x,cov) + sparsity*l1_norm(cov)

def l1_norm(cov):
    return sum(sum(abs(np.triu(cov,k=1))))

def likelihood(x,cov):
    '''
    input
    x: input unfold tensor [sample, dn]
    or
    x: input unfold tensor [sample, Dn, dn]
    cov: presicion matrix [dn, dn]
    '''
    costC=0
    if cov.shape==(1,1):
        return 0
    elif x.ndim==2:
        n,p=x.shape
        mu=np.mean(x,axis=0)
        det=np.linalg.det(cov) if np.linalg.det(cov)>0 else 1
        costC+=np.log(det)*(n/2) - np.log(2*np.pi)*(n*p/2)
        costC-=np.sum(np.diag(np.linalg.multi_dot([x-mu,cov,(x-mu).T])))/2
    else:
        n,p,d=x.shape
        det=np.linalg.det(cov) if np.linalg.det(cov)>0 else 1
        costC+=np.log(det)*(n*p/2) - np.log(2*np.pi)*(n*p*d/2)
        tmp=0
        for i in range(p):
            x_temp=x[:,i,:]
            mu=np.mean(x_temp,axis=0)
            tmp-=np.sum(np.diag(np.linalg.multi_dot([x_temp-mu,cov,(x_temp-mu).T])))/2
        # costC+=tmp/p
        costC+=tmp
        costC/=p
    return costC

def data_length_cost(index_list):
    costA=0
    for index in index_list:
        costA+=log_s(len(index))
    return costA

def log_s(x):
    #log*
    return 2. * np.log2(x) + 1.



class DMM():
    def __init__(self,sparsity=0.1,max_iter=100,save_result=True,save_dir='',verbose=True):
        self.sparsity=sparsity
        self.max_iter=max_iter
        self.save_result=save_result
        self.save_dir=save_dir
        self.verbose=verbose
        self.result=None
        self.time=0
        self.covariance_=None # covariance_=list(segment, ..., segment) n_seg, segment=list(tuple, tuple) n_dim, tuple=tuple(presicion, covariance)

    def fit(self,X,cp,gl_mode):
        starttime=time.time()
        self.print_verbose('Segmentation Start')
        new_cp,history_cp=self.segmentation(X,cp,gl_mode)
        self.print_verbose('Segmentation End')
        self.print_verbose('Classification Start')
        all_regime_objs,min_costT=self.viterbi_classification(X,new_cp,gl_mode)
        self.print_verbose('Classification End')
        self.time=time.time()-starttime
        self.result=all_regime_objs
        self.min_costT=min_costT
        if self.save_result:
            self.save(X,history_cp,history=True)

    def save(self,X,history_cp,history=None):
        util.save(self,X,history_cp,history)


    def segmentation(self,X,cp,gl_mode):
        '''
        input
        X: input tensor [time, d1, d2, ..., dn]
        cp: initial cut point, for example [5, 10, 15, ... , ]
        gl_mode: if 1 use each mean value [0, 1, ... , 0]
        '''
        sparsity=self.sparsity
        max_iter=self.max_iter

        length=X.shape[0]
        history_cp=[cp]
        self.print_verbose(f"first number of segment = {len(cp)+1}")
        iter=0
        while True:
            self.print_verbose(f"iteration {iter}")

            if len(cp)+1>2: # if there are 2 or more segments
                Single_CP=CutPoint(cp=cp,length=length)
                Single_Segments=Segments_GL(X,Single_CP.cp,gl_mode,sparsity,max_iter)
                Even_CP=Single_CP.even_cp()
                Even_Segments=Segments_GL(X,Even_CP.cp,gl_mode,sparsity,max_iter)
                Odd_CP=Single_CP.odd_cp()
                Odd_Segments=Segments_GL(X,Odd_CP.cp,gl_mode,sparsity,max_iter)
                self.covariance_=list([seg.Cov for seg in Single_Segments])

                index=0
                new_cp=np.empty(0,dtype=np.int32)
                while index+1<=Single_CP.num_seg-1: # compare MDL in chronological order
                    if index+1==Single_CP.num_seg-1: # if there left two segments at the end
                        if index%2==0:
                            End_CP=Even_CP
                            End_Segments=Even_Segments
                            end_index=int(index/2)
                        elif index%2==1:
                            End_CP=Odd_CP
                            End_Segments=Odd_Segments
                            end_index=int(index/2)+1
                        solo=segments_MDL([Single_Segments[index],Single_Segments[index+1]],X,gl_mode)
                        end=segments_MDL([End_Segments[end_index]],X,gl_mode)

                        if min(solo,end)==solo:
                            # print('solo')
                            point=[Single_CP.cp_value(index)]
                            index+=1
                        elif min(solo,end)==end:
                            # print('end')
                            point=[np.empty(0,dtype=np.int32)]
                            index+=1

                    else: # compare with three segments (mostly happens)
                        if index%2==0:
                            Left_CP=Even_CP
                            Right_CP=Odd_CP
                            Left_Segments=Even_Segments
                            Right_Segments=Odd_Segments
                            left_index=int(index/2)
                            right_index=int(index/2)+1
                        elif index%2==1:
                            Left_CP=Odd_CP
                            Right_CP=Even_CP
                            Left_Segments=Odd_Segments
                            Right_Segments=Even_Segments
                            left_index=int(index/2)+1
                            right_index=int(index/2)+1

                        solo=segments_MDL([Single_Segments[index],Single_Segments[index+1],Single_Segments[index+2]],X,gl_mode)
                        left=segments_MDL([Left_Segments[left_index],Single_Segments[index+2]],X,gl_mode)
                        right=segments_MDL([Single_Segments[index],Right_Segments[right_index]],X,gl_mode)

                        if min(solo,left,right)==solo:
                            # print('solo')
                            point=[Single_CP.cp_value(index)]
                            index+=1
                        elif min(solo,left,right)==left:
                            # print('left')
                            point=[Left_CP.cp_value(left_index)]
                            index+=2
                        elif min(solo,left,right)==right:
                            # print('right')
                            point=[Single_CP.cp_value(index)]
                            point+=[Right_CP.cp_value(right_index)]
                            index+=3
                    new_cp=np.append(new_cp,point)
                    new_cp=new_cp[new_cp!=Single_CP.length]
                self.print_verbose(f"number of segment = {len(new_cp)+1}")
                iter+=1

                if np.array_equal(cp,new_cp):
                    break
                cp=new_cp

            elif len(cp)+1==2: # if there left two segments (special case)
                Single_CP=CutPoint(cp=cp,length=length)
                Single_Segments=Segments_GL(X,cp,gl_mode,sparsity,max_iter)
                self.covariance_=list([seg.Cov for seg in Single_Segments])

                Whole_Segment=Segments_GL(X,[],gl_mode,sparsity,max_iter)

                solo=segments_MDL([Single_Segments[0],Single_Segments[1]],X,gl_mode)
                whole=segments_MDL([Whole_Segment[0]],X,gl_mode)
                new_cp=np.empty(0,dtype=np.int32)
                if min(solo,whole)==solo:
                    # print('solo')
                    new_cp=np.append(new_cp,[Single_CP.cp_value(0)])
                self.print_verbose(f"number of segment = {len(new_cp)+1}")
                break
            else: # if there is only one segment (impossible except initial value of cp=[])
                new_cp=np.empty(0,dtype=np.int32)
                self.print_verbose(f"number of segment = {len(new_cp)+1}")
                break
            history_cp.append(new_cp)

        return new_cp, history_cp

    def viterbi_classification(self,X,new_cp,gl_mode):
        from sklearn.cluster import KMeans

        def updateClusters(LLE_node_vals, switch_penalty=0):
            (T, num_clusters) = LLE_node_vals.shape
            future_cost_vals = np.zeros(LLE_node_vals.shape)

            # compute future costs
            for i in range(T-2, -1, -1):
                j = i+1
                indicator = np.zeros(num_clusters)
                future_costs = future_cost_vals[j, :]
                lle_vals = LLE_node_vals[j, :]
                for cluster in range(num_clusters):
                    total_vals = future_costs + lle_vals + switch_penalty
                    total_vals[cluster] -= switch_penalty
                    future_cost_vals[i, cluster] = np.min(total_vals)

            # compute the best path
            path = np.zeros(T)

            # the first location
            curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
            path[0] = curr_location

            # compute the path
            for i in range(T-1):
                j = i+1
                future_costs = future_cost_vals[j, :]
                lle_vals = LLE_node_vals[j, :]
                total_vals = future_costs + lle_vals + switch_penalty
                total_vals[int(path[i])] -= switch_penalty

                path[i+1] = np.argmin(total_vals)
            # return the computed path
            return path

        num_seg=len(new_cp)+1
        ndim = X.ndim - 1

        result_optRes=[Segment(X,gl_mode,index=np.arange(len(X)),id=0,Cov=self.covariance_[0],sparsity=self.sparsity,max_iter=self.max_iter)]
        min_costT=result_optRes[0].costT+3
        self.print_verbose(f"n_clusters ## 1\n")
        self.print_verbose(f"costT = {min_costT}")

        if num_seg==1:
            self.print_verbose("all at the same regime")
            return result_optRes, min_costT

        for k in range(2,num_seg+1):
            self.print_verbose(f"n_clusters ## {k}\n")
            num_of_clusters=k

            dim_kmean = 0
            for n in range(ndim):
                dim_kmean += self.covariance_[0][n][0].shape[0]**2
            cov_kmean = np.empty((num_seg,dim_kmean))
            for i in range(num_seg):
                taple_cov = self.covariance_[i]
                concat_cov = []
                for n in range(ndim):
                    concat_cov.append(taple_cov[n][0].flatten())
                cov_kmean[i,:] = np.hstack(concat_cov)
            init_cluster=KMeans(n_clusters=num_of_clusters,random_state=1).fit_predict(cov_kmean)

            clustered_points=init_cluster

            maxIters=10
            for iters in range(maxIters):
                self.print_verbose(f"iteration = {iters}\n")

                sample_clusters=[None for _ in range(num_of_clusters)]
                index_clusters=[None for _ in range(num_of_clusters)]
                for cluster in range(num_of_clusters):
                    id=np.empty(0,dtype=int)
                    for seg_id,seg_cluster in enumerate(clustered_points):
                        if seg_cluster==cluster:
                            _,id_temp=util.get_seq_at_index(X,new_cp,seg_id,return_index=True)
                            id=np.concatenate([id,id_temp],axis=0)
                    sample_clusters[cluster]=X[id]
                    index_clusters[cluster]=id

                optRes=[None for _ in range(num_of_clusters)]
                lle_segment_clusters=np.zeros([num_seg,num_of_clusters])
                for cluster in range(num_of_clusters):
                    optRes[cluster]=Segment(sample_clusters[cluster],gl_mode,index=index_clusters[cluster],id=cluster,sparsity=self.sparsity,max_iter=self.max_iter)
                    for seg_id,seg_cluster in enumerate(clustered_points):
                        x=util.get_seq_at_index(X,new_cp,seg_id)

                        lle_segment_clusters[seg_id,cluster]=0
                        for n in range(ndim):
                            x_unfold = sf_unfold(x, gl_mode, mode=n) # [sample, dn]
                            lle_segment_clusters[seg_id,cluster]+=data_coding_cost(x_unfold,optRes[cluster].Cov[n][0],optRes[cluster].sparsity)

                new_clustered_points=updateClusters(lle_segment_clusters,switch_penalty=0)

                if np.array_equal(clustered_points, new_clustered_points) or len(np.unique(new_clustered_points))!=num_of_clusters:
                    self.print_verbose("n_cluster converged!!\n")
                    break
                clustered_points=new_clustered_points

            costT=0
            n_cluster=len(np.unique(clustered_points))
            n_segment=0
            for cluster in range(n_cluster):
                costT+=optRes[cluster].costT
                n_segment+=len(optRes[cluster].index_list)
            costT+=log_s(n_segment)+log_s(n_cluster)+n_segment*log_s(n_cluster) #number of segments and cluster + assignment

            self.print_verbose(f"costT = {costT}")
            if min_costT>costT:
                result_optRes=optRes
                min_costT=costT
            else:
                self.print_verbose("viterbi break!!\n")
                self.print_verbose(f"number of regime = {len(result_optRes)}")
                self.print_verbose(f"minimum costT = {min_costT}\n")
                return result_optRes, min_costT
        return result_optRes, min_costT

    def print_verbose(self,word):
        if self.verbose:
            print(word)


class Param():
    def __init__(self,data_path,label_path,save_result=True,data_name='test',z_norm=True,window_z_norm=False,window=5,_sparsity=0.1,max_iter=100,_CF=32,_threshold=0.05,_alpha=1,evaluate=False,gl_mode=None):
        self.data_path=data_path
        self.label_path=label_path
        self.data_name=data_name
        self.save_result=save_result
        if save_result:
            base_dir=os.getcwd()
            result_dir=f'{base_dir}/result/{self.data_name}/'
            num=len(glob.glob(f'{result_dir}/*'))
            self.save_dir=f'{result_dir}/{num}'
            util.make_dir(self.save_dir)
        self.z_norm=z_norm
        self.window_z_norm=window_z_norm
        self.window=window
        self.sparsity=_sparsity
        self.max_iter=max_iter
        self._CF=_CF
        self._threshold=_threshold
        self._alpha=_alpha
        self.evaluate=evaluate
        self.gl_mode=gl_mode
        global CF
        CF=_CF
        global threshold
        threshold=_threshold
        global alpha
        alpha=_alpha
        global sparsity_gl
        sparsity_gl=_sparsity

def main(data_path,data_name,sparsity,window,label_path='',cf=32,z_norm=False,window_z_norm=False,evaluate=False):

    args=Param(data_path,label_path,save_result=True,data_name=data_name,z_norm=z_norm,window_z_norm=window_z_norm,window=window,_sparsity=sparsity,max_iter=100,_CF=cf,_threshold=0.05,_alpha=1,evaluate=evaluate)
    with open(f'{args.save_dir}/args.txt','w') as f:
        for arg in vars(args):
            f.write(f'{arg}:{vars(args)[arg]}\n')

    X,cp=util.data_import(args)
    X = np.squeeze(X)
    print('input X',X.shape)

    ndim = X.ndim-1
    gl_mode = []
    for n in range(ndim):
        gl_mode.append(1)

    ngl=DMM(sparsity=args.sparsity,max_iter=args.max_iter,save_result=args.save_result,save_dir=args.save_dir)
    ngl.fit(X,cp,gl_mode)

    if args.evaluate:
        transition=np.loadtxt(f'{args.save_dir}/transition.txt')
        y_true=np.loadtxt(args.label_path)
        accuracy, macro_f1, confusion_matrix=evaluation.evaluate_clustering_accuracy(y_true[:len(transition)],transition)
        print('macro_f1',macro_f1)
        print('accuracy',accuracy)
        np.savetxt(f'{args.save_dir}/f1.txt',[macro_f1],fmt='%.5e')
        np.savetxt(f'{args.save_dir}/accuracy.txt',[accuracy],fmt='%.5e')

def test(z_norm=False,window_z_norm=False,evaluate=True):
    data_path=''
    label_path=''
    data_name='test'
    sparsity=0.5
    cf=32
    window=4
    alpha=1

    args=Param(data_path,label_path,save_result=True,data_name=data_name,z_norm=z_norm,window_z_norm=window_z_norm,window=window,_sparsity=sparsity,max_iter=100,_CF=cf,_threshold=0.05,_alpha=alpha,evaluate=evaluate)
    with open(f'{args.save_dir}/args.txt','w') as f:
        for arg in vars(args):
            f.write(f'{arg}:{vars(args)[arg]}\n')

    X,cp=util.data_import(args)
    print('input X',X.shape)

    ndim = X.ndim-1
    gl_mode = []
    for n in range(ndim):
        gl_mode.append(1)

    ngl=DMM(sparsity=args.sparsity,max_iter=args.max_iter,save_result=args.save_result,save_dir=args.save_dir)
    ngl.fit(X,cp,gl_mode)

    util.save_gorgeous(args)

    if args.evaluate:
        transition=np.loadtxt(f'{args.save_dir}/transition.txt')
        y_true=np.loadtxt(args.label_path)
        accuracy, macro_f1, confusion_matrix=evaluation.evaluate_clustering_accuracy(y_true[:len(transition)],transition)
        print('macro_f1',macro_f1)
        print('accuracy',accuracy)
        np.savetxt(f'{args.save_dir}/f1.txt',[macro_f1],fmt='%.5e')
        np.savetxt(f'{args.save_dir}/accuracy.txt',[accuracy],fmt='%.5e')

if __name__=='__main__':
    print('DMM')
    test()
