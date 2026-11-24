# BiGCN — modernized edition
Based on Tian Bian et al., Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks (AAAI 2020). Original source and README are retained in legacy/ for reference, not as runnable entry points.

## Windows setup (Python 3.11 or 3.12)
Extract the ZIP, open PowerShell in this folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe bigcn.py --smoke --device cpu
```

For RTX 5080, install a Blackwell-compatible PyTorch build (CUDA 12.8 or newer) using the official selector at https://pytorch.org/get-started/locally/ before running training. For example, install torch from https://download.pytorch.org/whl/cu128 using pip's --index-url option. Update NVIDIA drivers. GPU execution has not been tested here.

## Training
```powershell
.\.venv\Scripts\python.exe bigcn.py --dataset Twitter15 --epochs 20 --batch-size 16
```
Use Twitter16 or Weibo as the dataset argument. Reduce batch size if memory runs out. Default is 200 epochs, one iteration, five folds. Each fold uses three training folds, one validation fold and one held-out test fold. This differs from the original experiment protocol and is not a claim of reproducing published scores. Checkpoints are saved in checkpoints/. Smoke mode uses a tiny subset and one epoch, not a meaningful accuracy benchmark. Weibo is read directly from its ZIP, which is not bundled in this compact edition. Run `python download_weibo.py` or copy `data/Weibo/weibotree.txt.zip` from the original repository into the same path. The two directional GCN branches, root feature concatenation, mean pooling and edge dropout are retained. No raw-text prediction interface or pretrained weights are supplied.

## Verification
CPU smoke tests with PyTorch 2.10.0: Twitter15 (1470 graphs), Twitter16 (808), Weibo (4527). One epoch on tiny subsets; accuracy 25%, 25%, 0% respectively, not full benchmarks. GPU and Windows launcher not tested.
Original repository: https://github.com/TianBian95/BiGCN
