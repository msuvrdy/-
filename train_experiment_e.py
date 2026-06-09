"""Experiment E: DG LOSO, 87% version"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse, os
from sklearn.metrics import confusion_matrix, classification_report

FAULT2ID = {'H_H':0,'R_U':1,'R_M':2,'S_W':3,'V_U':4,'B_R':5,'K_A':6,'F_B':7}
ID2FAULT = {v:k for k,v in FAULT2ID.items()}

class AllWindowDataset(Dataset):
    def __init__(self, batch_dir, speed_ids, load_id, split='full', indices=None, num_batches=8):
        fo = ['B_R','F_B','H_H','K_A','R_M','R_U','S_W','V_U']
        wpf = {'train':132,'val':29,'test':40}
        fi=[]
        for bi in range(num_batches):
            bl=0 if bi<num_batches//2 else 1; fs=(bi%(num_batches//2))*2
            for fj in range(2):
                f=fo[fs+fj]
                for sp in [1,2,3,4]: fi.append({'speed':sp,'load':bl,'fault':f})
        ml,tl,ll,sl=[],[],[],[]
        for i,fi_ in enumerate(fi):
            if fi_['speed'] not in speed_ids or fi_['load']!=load_id: continue
            bi=i//8; foi=i%8; d=np.load(f"{batch_dir}/batch_{bi:02d}.npz")
            for seg,tk,lk in [('train','tst','tlt'),('val','vst','vlt'),('test','est','elt')]:
                wp=wpf[seg]; s,e=foi*wp,(foi+1)*wp
                ml.append(d[tk][s:e])
                tl.append(d['tvt'if seg=='train'else('vvt'if seg=='val'else'evt')][s:e])
                ll.append(d[lk][s:e]); sl.append(np.full(e-s,fi_['speed'],dtype=np.int32))
        am=np.concatenate(ml,0); at=np.concatenate(tl,0); al=np.concatenate(ll,0); asp=np.concatenate(sl,0)
        if indices is not None and split!='full':
            am=am[indices]; at=at[indices]; al=al[indices]; asp=asp[indices]
        self.mr=am; self.tr=at; self.lr=al; self.sr=asp
        st=np.load(f"{batch_dir}/../norm_stats.npz")
        nf=int(st['stft_mean'].size//4)
        self.mm=torch.from_numpy(st['stft_mean']).float().view(4,nf,1)
        self.ms=torch.from_numpy(st['stft_std']).float().view(4,nf,1)+1e-8
        self.tmin=at.min(); self.tmax=at.max()
    def __len__(self): return len(self.lr)
    def __getitem__(self,idx):
        mel=torch.from_numpy(self.mr[idx]).float(); mel=(mel-self.mm)/self.ms
        tv=self.tr[idx]; temp=(tv-self.tmin)/(self.tmax-self.tmin+1e-8)
        temp=torch.tensor([temp],dtype=torch.float32)
        label=torch.tensor(self.lr[idx],dtype=torch.long)
        speed=torch.tensor(self.sr[idx],dtype=torch.long)
        return mel,temp,label,speed

class ChannelAttention(nn.Module):
    def __init__(self,ch,red=8):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(ch,ch//red,bias=False),nn.ReLU(),nn.Linear(ch//red,ch,bias=False),nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.shape; y=x.view(b,c,-1).mean(-1,keepdim=True).view(b,c)
        return x*self.fc(y).view(b,c,1,1)

class SpatialAttention(nn.Module):
    def __init__(self): super().__init__(); self.conv=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        a=x.mean(1,keepdim=True); m=x.max(1,keepdim=True)[0]
        return x*torch.sigmoid(self.conv(torch.cat([a,m],1)))

class CBAM(nn.Module):
    def __init__(self,ch,red=8): super().__init__(); self.ca=ChannelAttention(ch,red); self.sa=SpatialAttention()
    def forward(self,x): return self.sa(self.ca(x))

class LightCNN_CBAM_SupCon(nn.Module):
    def __init__(self,num_classes=8,ic=4,pd=128):
        super().__init__()
        self.c1=nn.Sequential(nn.Conv2d(ic,32,5,2,2),nn.BatchNorm2d(32),nn.ReLU())
        self.c2=nn.Sequential(nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU())
        self.c3=nn.Sequential(nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU())
        self.c4=nn.Sequential(nn.Conv2d(128,256,3,2,1),nn.BatchNorm2d(256),nn.ReLU())
        self.cbam=CBAM(256); self.cbam2=CBAM(64); self.cbam3=CBAM(128)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.tmlp=nn.Sequential(nn.Linear(1,16),nn.ReLU())
        fd=256+16
        self.cls=nn.Sequential(nn.Linear(fd,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.5),nn.Linear(128,num_classes))
        self.proj=nn.Sequential(nn.Linear(fd,256),nn.ReLU(),nn.Linear(256,pd))
    def ff(self,m,t):
        x=self.c1(m); x=self.c2(x); x=self.c3(x); x=self.c4(x)
        x=self.cbam(x); f=self.pool(x).flatten(1)
        return torch.cat([f,self.tmlp(t)],1)
    def classify(self,f): return self.cls(f)
    def project(self,f): return F.normalize(self.proj(f),1)
    def forward(self,m,t,re=False):
        f=self.ff(m,t); l=self.classify(f)
        if re: return l,self.project(f)
        return l

def mixup_data(x,y,alpha=0.2):
    if alpha>0: lam=np.random.beta(alpha,alpha)
    else: lam=1
    idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx],y,y[idx],lam

def mixup_criterion(crit,pred,ya,yb,lam):
    return lam*crit(pred,ya)+(1-lam)*crit(pred,yb)

def supcon_loss(features,labels,temp=0.07):
    N=features.size(0); sim=torch.matmul(features,features.T)/temp
    pm=(labels.unsqueeze(0)==labels.unsqueeze(1)).float(); pm.fill_diagonal_(0)
    sm=sim.clone(); sm[torch.eye(N,dtype=bool,device=sim.device)]=float('-inf')
    lse=torch.logsumexp(sm,1); np_=pm.sum(1); ps=(sim*pm).sum(1); mp=ps/np_.clamp(min=1)
    loss=(-mp+lse); hp=np_>0; return loss[hp].mean()

def train_epoch(model,loader,opt,cc,dev,lsc=0.2,mw=6,nm=2):
    model.train(); tl=0
    for mel,temp,labels,speeds in loader:
        mel,temp,labels=mel.to(dev),temp.to(dev),labels.to(dev); B=mel.size(0)
        mel_m=mel.clone()
        for i in range(B):
            n=torch.randint(1,nm+1,(1,)).item()
            for _ in range(n):
                w=torch.randint(mw//2,mw+1,(1,)).item()
                f0=torch.randint(0,mel.size(2)-w,(1,)).item()
                mel_m[i,:,f0:f0+w,:]=mel[i].mean()
        mb=torch.cat([mel,mel_m],0); tb=torch.cat([temp,temp],0)
        lb=torch.cat([labels,labels],0)
        lob,eb=model(mb,tb,re=True)
        sl=supcon_loss(eb,lb); cl=cc(lob[:B],labels)
        loss=cl+lsc*sl
        mm,la,lb_,lam=mixup_data(mel,labels)
        lm=model(mm,temp); ml_=mixup_criterion(cc,lm,la,lb_,lam)
        opt.zero_grad(); (loss+ml_).backward(); opt.step()
        tl+=loss.item()+ml_.item()
    return tl/len(loader)

@torch.no_grad()
def evaluate(model,loader,dev):
    model.eval(); ap,al=[],[]; c,t=0,0; cr=nn.CrossEntropyLoss(); tl=0
    for mel,temp,labels,speeds in loader:
        mel,temp,labels=mel.to(dev),temp.to(dev),labels.to(dev)
        o=model(mel,temp); tl+=cr(o,labels).item()
        _,p=o.max(1); c+=p.eq(labels).sum().item(); t+=labels.size(0)
        ap.extend(p.cpu().numpy()); al.extend(labels.cpu().numpy())
    return tl/len(loader),100.*c/t,ap,al

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',type=str,required=True)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--device',type=str,default='cuda')
    p.add_argument('--lambda_sc',type=float,default=0.12); p.add_argument('--mask_width',type=int,default=3)
    p.add_argument('--n_masks_max',type=int,default=2); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output_dir',type=str,default='./results_exp_e')
    args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir,exist_ok=True); bd=args.data_dir
    folds=[{'n':'Fold 1 (15Hz test)','ts':[2,3,4],'t':1,'l':0},
           {'n':'Fold 2 (30Hz test)','ts':[1,3,4],'t':2,'l':0},
           {'n':'Fold 3 (45Hz test)','ts':[1,2,4],'t':3,'l':0},
           {'n':'Fold 4 (60Hz test)','ts':[1,2,3],'t':4,'l':0}]
    results=[]
    for fold in folds:
        print(f"\n{'='*60}\n{fold['n']}\nTrain: {fold['ts']}->Test: {fold['t']}\n{'='*60}")
        pool=AllWindowDataset(bd,fold['ts'],fold['l'],split='full')
        nt=len(pool); idx=np.random.permutation(nt); ntr=int(nt*0.8)
        tr=AllWindowDataset(bd,fold['ts'],fold['l'],split='train',indices=idx[:ntr])
        va=AllWindowDataset(bd,fold['ts'],fold['l'],split='val',indices=idx[ntr:])
        te=AllWindowDataset(bd,[fold['t']],fold['l'],split='full')
        print(f"Tr: {len(tr)} Va: {len(va)} Te: {len(te)}")
        trl=DataLoader(tr,args.batch_size,shuffle=True); val=DataLoader(va,args.batch_size,shuffle=False)
        tel=DataLoader(te,args.batch_size,shuffle=False)
        model=LightCNN_CBAM_SupCon(num_classes=8,ic=4).to(dev)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
        opt=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
        cc=nn.CrossEntropyLoss(label_smoothing=0.1)
        bv,pt,pc=0,15,0
        for ep in range(args.epochs):
            trl_=train_epoch(model,trl,opt,cc,dev,lsc=args.lambda_sc,mw=args.mask_width,nm=args.n_masks_max)
            vl,va_,_,_=evaluate(model,val,dev); sch.step()
            if va_>bv: bv=va_; pc=0; torch.save(model.state_dict(),f"{args.output_dir}/best_{fold['t']}.pth")
            else: pc+=1
            if ep%5==0 or ep<3: print(f"  Ep{ep:2d}: TrL={trl_:.4f} VL={vl:.4f} VA={va_:.2f}%")
            if pc>=pt: print(f"  Stop ep{ep}"); break
        print(f"\nBest VA: {bv:.2f}%")
        model.load_state_dict(torch.load(f"{args.output_dir}/best_{fold['t']}.pth"))
        _,ta,tp,tl=evaluate(model,tel,dev)
        print(f"TEST: {ta:.2f}%")
        results.append({'f':fold['n'],'t':fold['t'],'ta':ta,'tp':tp,'tl':tl})
        fn=[ID2FAULT[i] for i in range(8)]
        for c in range(8):
            m=np.array(tl)==c
            if m.sum()>0: print(f"  {ID2FAULT[c]:5s}: {100*np.sum((np.array(tp)==c)&m)/m.sum():.1f}%")
        print("\n"+classification_report(tl,tp,target_names=fn,digits=3,zero_division=0))
        cm=confusion_matrix(tl,tp)
        print("  "+"".join(f"{n:>6s}"for n in fn))
        for i,n in enumerate(fn): print(f"{n:5s}"+"".join(f"{cm[i,j]:6d}"for j in range(8)))
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    accs=[r['ta']for r in results]
    for r in results: print(f"  {r['f']}: {r['ta']:.2f}%")
    print(f"  Avg: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
    np.savez(f"{args.output_dir}/results.npz",results=results,accs=accs)

if __name__=='__main__': main()
