"""Restore the original Weibo archive with SHA-256 verification."""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
URL = 'https://raw.githubusercontent.com/TianBian95/BiGCN/5779f2cf86a4649651f2466b43c73caa6ed7463e/data/Weibo/weibotree.txt.zip'
SHA = 'd8534b562f2354ecf995f0a94aac39f389d58cd962799bc89b13e238da58f883'
if __name__ == '__main__':
    dest = Path(__file__).resolve().parent / 'data/Weibo/weibotree.txt.zip'
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.download')
    try:
        urlretrieve(URL, temp)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA:
            raise ValueError('Dataset checksum mismatch')
        temp.replace(dest)
        print('Weibo dataset verified and installed.')
    finally:
        temp.unlink(missing_ok=True)
