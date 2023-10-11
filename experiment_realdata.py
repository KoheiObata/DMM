import numpy as np
import pandas as pd
import glob
import datetime


from dateutil.relativedelta import relativedelta
from src import util
from src.model import Param, DMM


def get_air(DATADIR,st,ed):
    file_list=glob.glob(f'{DATADIR}/*')
    df=pd.DataFrame()
    for file_path in file_list:
        location=file_path.split('/')[-1].split('_')[2]
        print(location)
        df_temp=pd.read_csv(file_path)
        df_temp['date']=df_temp['year'].astype('str')
        df_temp['date']=df_temp['date'].str.cat([df_temp['month'].astype('str'),df_temp['day'].astype('str')],sep='-')
        df_temp['date']=pd.to_datetime(df_temp['date'],format='%Y-%m-%d')

        df_temp=df_temp.loc[(df_temp['date']>=st) & (df_temp['date']<ed)]
        df_temp=df_temp.set_index('date')
        df_temp=df_temp[['PM2.5','PM10','SO2','NO2','CO','O3','station']]
        df_temp=df_temp.interpolate()
        df_temp=df_temp[::24] # hourly data to daily data.
        df=pd.concat([df,df_temp],axis=0)
    return df, file_list

def get_cp_air(df,interval):
    station = df['station'].unique()[0]
    df = df.loc[df['station']==station]
    st=df.index[0]
    ed=df.index[-1]
    cp=[]
    idx=0
    while st<ed:
        idx+=len(df.loc[(df.index>=st)&(df.index<st+interval)])
        cp.append(idx)
        st+=interval
    return np.array(cp[:-1])


def import_air(DATADIR, st, ed, interval):
    df, file_list = get_air(DATADIR, st, ed)
    cp = get_cp_air(df, interval)

    df = df.reset_index(drop=True)

    data=np.empty((int(len(df)/len(file_list)),len(df.columns)-1,len(file_list)))
    for i,file_path in enumerate(file_list):
        location=file_path.split('/')[-1].split('_')[2]
        X=df.loc[df['station']==location].drop(['station'],axis=1)
        X,_,_=util.norm_X_at_cp(X,cp)
        # X,_,_=util.norm_X_at_cp(X,[])
        data[:,:,i]=X
    return data, cp


def get_nameandtime(DATADIR):
    dataname=DATADIR.split('/')[-1]
    if dataname=='covid_flu':
        area='country'
        st=datetime.datetime(year=2013,month=1,day=1)
        ed=datetime.datetime(year=2023,month=1,day=1)
    elif dataname in ('gafam'):
        area='country'
        st=datetime.datetime(year=2015,month=1,day=1)
        ed=datetime.datetime(year=2020,month=1,day=1)
    elif dataname in ('sweets','commerce','vod'):
        area='region'
        st=datetime.datetime(year=2015,month=1,day=1)
        ed=datetime.datetime(year=2020,month=1,day=1)

    interval=relativedelta(months=1)

    if area=='country':
        top_10=['US','CN','JP','DE','IN','GB','FR','BR','IT','CA']
    elif area=='region':
        top_10=['US-CA','US-TX','US-FL','US-NY','US-PA','US-IL','US-OH','US-GA','US-NC','US-MI']
    return st, ed, interval, top_10


def load_google_data(data_dir,st,ed,interval,top_10):
    file_list=glob.glob(f'{data_dir}/*')
    df=pd.DataFrame()
    dates=pd.date_range(start=st, end=ed-interval, freq='D')
    # dates=pd.DataFrame(dates, columns={'date'})
    dates=pd.DataFrame(dates)
    dates=dates.rename(columns={0:'date'})
    for filepath in file_list:
        name=filepath.split('/')[-1].replace('.csv','')
        df_temp=pd.read_csv(filepath)
        df_temp=df_temp[[*top_10,'date']]
        df_temp['name']=name
        df_temp['date']=pd.to_datetime(df_temp['date'])
        df_temp=df_temp.loc[(df_temp['date']>=st)&(df_temp['date']<ed)]
        df_temp=pd.merge_asof(dates, df_temp, on='date')
        df_temp=df_temp.set_index('date')
        df_temp=df_temp[~df_temp.index.duplicated(keep='first')]
        print(name,len(df_temp))
        df=pd.concat([df,df_temp],axis=0)
    return df

def get_googletrend(DATADIR,st,ed,top_10):
    file_path=glob.glob(f'{DATADIR}/*')
    df=pd.DataFrame()
    for filepath in file_path:
        name=filepath.split('/')[-1].replace('.csv','')
        df_temp=pd.read_csv(filepath)
        df_temp=df_temp[[*top_10,'date']]
        df_temp['name']=name
        df_temp['date']=pd.to_datetime(df_temp['date'])
        df_temp=df_temp.loc[(df_temp['date']>=st)&(df_temp['date']<ed)]
        df_temp=df_temp.set_index('date')
        df_temp=df_temp[~df_temp.index.duplicated(keep='first')]
        print(name,len(df_temp))
        df=pd.concat([df,df_temp],axis=0)
    return df

def get_cp_googletrend(df,interval):
    st=df.index[0]
    ed=df.index[-1]
    cp=[]
    idx=0
    while st<ed:
        idx+=len(df.loc[(df.index>=st)&(df.index<st+interval)])
        cp.append(idx)
        st+=interval
    return np.array(cp[:-1])

def import_googletrend(DATADIR):
    st, ed, interval, top_10 = get_nameandtime(DATADIR)
    if DATADIR.split('/')[-1] in ('apparel','music','SNS','sweets','commerce','vod','facilities'):
        df=load_google_data(DATADIR,st,ed,interval,top_10)
    else:
        df=get_googletrend(DATADIR,st,ed,top_10)
    name_list=df['name'].unique()
    df_temp=df.loc[df['name']==name_list[0]]
    cp=get_cp_googletrend(df_temp,interval)


    data=np.empty((int(len(df)/len(name_list)),len(name_list),len(top_10)))
    for i,name in enumerate(name_list):
        X=df.loc[df['name']==name].drop(['name'],axis=1)
        X,_,_=util.norm_X_at_cp(X,cp)
        data[:,i,:]=X
    return data, cp



def import_data(name):
    basedir='.'
    if name=='air':
        #X=(time,key,location)
        DATADIR=f'{basedir}/data/PRSA_Data_20130301-20170228'
        st=datetime.datetime(year=2013,month=3,day=1)
        ed=datetime.datetime(year=2017,month=3,day=1)
        interval=relativedelta(months=1)
        data, cp = import_air(DATADIR, st, ed, interval)

    elif name in ('covid_flu','gafam','vod','commerce','sweets'):
        DATADIR=f'{basedir}/data/google/{name}'
        data, cp = import_googletrend(DATADIR)

    return data, cp, DATADIR



names=['commerce' ,'vod' ,'sweets','covid_flu' ,'gafam' ,'air']

for name in names:
    data,cp,DATADIR = import_data(name)

    data_path=DATADIR
    label_path=''
    data_name=name
    z_norm=False
    window_z_norm=True
    gl_mode=list(np.ones(data.ndim - 1))
    for sparsity in [0.5]:
        alpha=1
        args=Param(data_path,label_path,save_result=True,data_name=data_name,z_norm=z_norm,window_z_norm=window_z_norm,_sparsity=sparsity,_alpha=alpha,evaluate=False,gl_mode=gl_mode)
        with open(f'{args.save_dir}/args.txt','w') as f:
            for arg in vars(args):
                f.write(f'{arg}:{vars(args)[arg]}\n')

        print('input data',data.shape)

        ddnf=DMM(sparsity=args.sparsity,max_iter=args.max_iter,save_result=args.save_result,save_dir=args.save_dir)
        ddnf.fit(data,cp,gl_mode)
