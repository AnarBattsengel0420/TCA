import sys, time
sys.path.insert(0,'.')
from App import *

docs, skip = load_documents(r'D:\sample_data')
print(f'File: {len(docs)}')

fc = build_chunks_parallel(docs)
nc = sum(len(v) for v in fc.values())
print(f'Chunk: {nc}')

at = [c['text'] for cs in fc.values() for c in cs]
bm = BM25Index(); bm.build(at)

eng = EmbedEngine(); eng._load()
print(f'Backend: {eng.backend}')

t=time.perf_counter()
eng.encode(at[:200])
r=200/max(time.perf_counter()-t,0.001)
est=nc/r
print(f'Embed: {r:.0f} chunk/s | est: {est:.0f}s')
total=est+10
print(f'Total est: {total:.0f}s')
if total<=700: print('OK')
else: print(f'Over: +{total-700:.0f}s')
