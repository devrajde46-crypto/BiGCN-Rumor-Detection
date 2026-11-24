"""Modern PyTorch-only implementation of the original BiGCN architecture."""
import argparse, random, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parent
LABELS = {'news':0,'non-rumor':0,'false':1,'true':2,'unverified':3}

def load_events(name):
    folder = ROOT / 'data' / name
    weibo = name == 'Weibo'
    labels = {}
    with (folder / ('weibo_id_label.txt' if weibo else name+'_label_All.txt')).open(encoding='utf8') as f:
        for line in f:
            p = line.split() if weibo else line.rstrip('\n').split('\t')
            labels[p[0] if weibo else p[2]] = int(p[1]) if weibo else LABELS[p[0].lower()]
    trees = defaultdict(dict)
    def parse(lines):
        for line in lines:
            if isinstance(line, bytes): line = line.decode('utf8')
            p = line.rstrip('\r\n').split('\t')
            if p[0] in labels:
                trees[p[0]][int(p[2])] = (p[1], p[3 if weibo else 5] if len(p)> (3 if weibo else 5) else '')
    if weibo:
        with zipfile.ZipFile(folder/'weibotree.txt.zip') as z, z.open('weibotree.txt') as f: parse(f)
    else:
        with (folder/'data.TD_RvNN.vol_5000.txt').open(encoding='utf8') as f: parse(f)
    events = []
    for eid, nodes in trees.items():
        roots = [i for i,(p,_) in nodes.items() if p=='None']
        if len(nodes)<2 or len(roots)!=1: continue
        ids = {n:i for i,n in enumerate(sorted(nodes))}
        edges=[]; rows=[]; cols=[]; vals=[]
        for n,(parent,vec) in nodes.items():
            if parent!='None' and int(parent) in ids: edges.append((ids[int(parent)],ids[n]))
            for pair in vec.split():
                a,b=pair.split(':'); a=int(a)-int(weibo)
                if 0<=a<5000: rows.append(ids[n]); cols.append(a); vals.append(float(b))
        events.append((len(ids),rows,cols,vals,edges,ids[roots[0]],labels[eid]))
    return events

def collate(events, drop=0):
    xs=[]; td=[]; bu=[]; roots=[]; batches=[]; ys=[]; offset=0
    for k,(n,r,c,v,edges,root,y) in enumerate(events):
        x=torch.zeros(n,5000); x[r,c]=torch.tensor(v,dtype=torch.float32); xs.append(x)
        for dest, reverse in [(td,False),(bu,True)]:
            chosen=random.sample(edges,int(len(edges)*(1-drop))) if drop else edges
            dest.extend((b+offset,a+offset) if reverse else (a+offset,b+offset) for a,b in chosen)
        roots.append(root+offset); batches.extend([k]*n); ys.append(y); offset+=n
    return (torch.cat(xs),torch.tensor(td,dtype=torch.long).reshape(-1,2).T.contiguous(),torch.tensor(bu,dtype=torch.long).reshape(-1,2).T.contiguous(),torch.tensor(roots),torch.tensor(batches),torch.tensor(ys))

class GCN(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.linear=nn.Linear(a,b,bias=False); self.bias=nn.Parameter(torch.zeros(b))
    def forward(self,x,edge):
        n=x.shape[0]; src,dst=edge; keep=src!=dst; loop=torch.arange(n,device=x.device)
        src=torch.cat((src[keep],loop)); dst=torch.cat((dst[keep],loop))
        deg=x.new_zeros(n).index_add_(0,dst,x.new_ones(dst.numel())).rsqrt()
        h=self.linear(x); out=h.new_zeros(h.shape).index_add_(0,dst,h[src]*(deg[src]*deg[dst])[:,None]); return out+self.bias

class Branch(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1=GCN(5000,64); self.conv2=GCN(5064,64)
    def forward(self,x,edge,roots,batch):
        h=self.conv1(x,edge); z=F.dropout(F.relu(torch.cat((h,x[roots][batch]),1)),training=self.training)
        z=torch.cat((F.relu(self.conv2(z,edge)),h[roots][batch]),1)
        return z.new_zeros((len(roots),128)).index_add_(0,batch,z)/torch.bincount(batch,minlength=len(roots))[:,None]

class BiGCN(nn.Module):
    def __init__(self,classes):
        super().__init__(); self.td=Branch(); self.bu=Branch(); self.fc=nn.Linear(256,classes)
    def forward(self,x,td,bu,roots,batch):
        return self.fc(torch.cat((self.bu(x,bu,roots,batch),self.td(x,td,roots,batch)),1))

def folds(events,seed):
    rng=random.Random(seed); groups=defaultdict(list); result=[[] for _ in range(5)]
    for e in events: groups[e[-1]].append(e)
    for group in groups.values():
        rng.shuffle(group)
        for i,e in enumerate(group): result[i%5].append(e)
    return result

def evaluate(model,events,device,batch_size,classes):
    model.eval(); loss=0; predictions=[]; targets=[]
    with torch.no_grad():
        for batch in DataLoader(events,batch_size=batch_size,collate_fn=collate):
            *x,y=[t.to(device) for t in batch]; logits=model(*x); loss+=F.cross_entropy(logits,y,reduction='sum').item(); predictions.extend(logits.argmax(1).cpu().tolist()); targets.extend(y.cpu().tolist())
    p=np.array(predictions); y=np.array(targets); scores=[]
    for c in range(classes):
        tp=((p==c)&(y==c)).sum(); den=(p==c).sum()+(y==c).sum(); scores.append(float(2*tp/den) if den else 0.)
    return loss/len(y),float((p==y).mean()),scores

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['Twitter15','Twitter16','Weibo'],default='Twitter15'); ap.add_argument('--epochs',type=int,default=200); ap.add_argument('--iterations',type=int,default=1); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='auto'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    if min(args.epochs,args.iterations,args.batch_size)<1: ap.error('positive values required')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device); torch.manual_seed(args.seed); random.seed(args.seed)
    if device.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    classes=2 if args.dataset=='Weibo' else 4; events=load_events(args.dataset); print(f'{args.dataset}: {len(events)} usable graphs; device={device}',flush=True)
    if args.smoke:
        groups=defaultdict(list)
        for e in events: groups[e[-1]].append(e)
        events=[e for g in groups.values() for e in sorted(g,key=lambda e:e[0])[:5]]; args.epochs=1; args.iterations=1
    output=ROOT/'checkpoints'; output.mkdir(exist_ok=True); results=[]
    for run in range(args.iterations):
        split=folds(events,args.seed+run)
        for k in range(1 if args.smoke else 5):
            test=split[k]; val=split[(k+1)%5]; train=[e for j,g in enumerate(split) if j not in (k,(k+1)%5) for e in g]
            model=BiGCN(classes).to(device); opt=torch.optim.Adam([{'params':list(model.td.parameters())+list(model.fc.parameters())},{'params':model.bu.parameters(),'lr':0.0001}],lr=0.0005,weight_decay=1e-4)
            best=float('inf'); stale=0; path=output/f'{args.dataset}_run{run}_fold{k}.pt'
            for epoch in range(args.epochs):
                model.train()
                for batch in DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=lambda e:collate(e,.2)):
                    *x,y=[t.to(device) for t in batch]; opt.zero_grad(); loss=F.cross_entropy(model(*x),y); loss.backward(); opt.step()
                loss,acc,_=evaluate(model,val,device,args.batch_size,classes); print(f'Fold {k+1} epoch {epoch+1}: validation loss={loss:.4f}, accuracy={acc:.4f}',flush=True)
                if loss<best: best=loss; stale=0; torch.save(model.state_dict(),path)
                else: stale+=1
                if stale>=10: break
            model.load_state_dict(torch.load(path,map_location=device,weights_only=True)); _,acc,f1=evaluate(model,test,device,args.batch_size,classes); results.append([acc,*f1]); print(f'Test accuracy={acc:.4f}; class F1={f1}',flush=True)
    print('Mean test accuracy and class F1:',np.mean(results,axis=0))

if __name__=='__main__': main()
