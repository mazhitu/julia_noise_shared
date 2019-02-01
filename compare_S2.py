import pyasdf
import h5py
import numpy as np
from matplotlib import pyplot as plt

CCFDIR = '/Users/zma/chengxin/KANTO/CCF_python'
py_allnames=[]
py_fid=pyasdf.ASDFDataSet(CCFDIR+"/2012_01_01.h5",mode='r')
for source in py_fid.auxiliary_data.list():
    for receiver in py_fid.auxiliary_data[source].list():
        py_allnames.append((source,receiver))
print("from python file #=",len(py_allnames))

CCFDIR = '/Users/zma/chengxin/KANTO/CCF'
jul_fid=h5py.File(CCFDIR+"/test.h5",'r')
jul_allnames=list(jul_fid.keys())
print("from julia file #=",len(jul_allnames))

id=887
source=py_allnames[id][0].replace('s','_')[2:]
receiver=py_allnames[id][1][2:]
name=source+'_'+receiver
if name not in jul_allnames:
    name=receiver+'_'+source
    print("flipping")
    py_dat=py_fid.auxiliary_data[py_allnames[id][0]][py_allnames[id][1]].data[:][::-1]
else:
    py_dat=py_fid.auxiliary_data[py_allnames[id][0]][py_allnames[id][1]].data[:]

print("showing:",name)
print(name in jul_allnames)
jul_dat=jul_fid[name][:]

maxlag = 800
downsamp_freq=20
dt=1/downsamp_freq

jul_t=np.linspace(-maxlag,maxlag,len(jul_dat))
py_t=np.arange(-maxlag-dt,maxlag,dt)
plt.plot(jul_t,jul_dat);plt.plot(py_t,py_dat,'.')
plt.show()