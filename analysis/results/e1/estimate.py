# 저자 E1 Base 지도(256px)와 타일별 클래스 픽셀 수가 가장 비슷해지는 θ를 찾음 (Nelder-Mead, 추론 약 45회)
import sys, json, types, numpy as np
sys.argv=['x']; sys.path.insert(0,'/home/claude/rsm-calib')
import run
from data import Evaluator, confusion
from scipy.optimize import minimize
E='/tmp/claude-0/e1'; O='/tmp/claude-0/e1est'
a=types.SimpleNamespace(img_dir=E+'/img',msk_dir=E+'/msk',img_glob='LC_*.tif',msk_glob='LC_*.tif',include=None,exclude=None,tiles=None,limit=None,label_map='/home/claude/rsm-calib/labelmap_aihub_e1.json',n_classes=19,bands=[0,1,2],checkpoint='/mnt/user-data/uploads/models/rgb15/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth',encoder='resnet34',arch='Unet')
imgs,msks=run._load(a); ev=Evaluator(run._setup_model(a),imgs,19)
au=json.load(open(O+'/author_base_counts.json')); A=np.array([r[1:] for r in au],float)
K=[0,4,5,6,17]; g=msks[:,::2,::2]; valid=np.isin(g,K)
log=open(O+'/trace.csv','w'); log.write('R,G,B,Rs,Gs,Bs,loss,bld,water,conif,decid,green\n'); best=[1e18,None]
def f(t):
    p=ev.predict(np.asarray(t,float)); q=p[:,::2,::2]
    C=np.stack([((q==k)&valid).sum((1,2)) for k in K],1)
    loss=np.abs(C-A).sum()/A.sum()
    cm=confusion(p,msks,19); iou=[100*cm[k,k]/(cm[k,:].sum()+cm[:,k].sum()-cm[k,k]) for k in K]
    log.write(','.join(f'{v:.4f}' for v in list(t)+[loss]+iou)+'\n'); log.flush()
    if loss<best[0]: best[:]=[loss,list(t)]
    return loss
x0=np.array([115.03,117.30,108.04,64.65,59.15,59.37])
minimize(f,x0,method='Nelder-Mead',options=dict(maxfev=45,initial_simplex=np.vstack([x0]+[x0*(1+0.08*np.eye(6)[i]) for i in range(6)])))
json.dump(dict(loss=best[0],theta=best[1]),open(O+'/best.json','w'),indent=1)
