import sys, time
sys.path.insert(0,'.')
from App import *

t0=time.perf_counter()
store=HybridStore()
docs,_=load_documents(r'D:\sample_data')
import hashlib
fid=hashlib.md5(r'D:\sample_data'.encode()).hexdigest()
n=store.build(docs, folder_id=fid)
store.save(_index_base('sample_data'))
t=time.perf_counter()-t0
print(f'DONE: {n} chunk | {t:.0f}s ({t/60:.1f} min)')
