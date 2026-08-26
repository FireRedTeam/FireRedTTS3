"""Speaker-similarity check: cosine sim between each generated clone and its
reference, using the CAM++ extractor that ships with the model."""
import os
import sys
import torch
import torchaudio
import torch.nn.functional as F

from fireredtts3.campp.campp import CamppEmbedding

refs_dir, out_dir = sys.argv[1], sys.argv[2]
spk = CamppEmbedding(os.path.join('pretrained_models', 'campp', 'campplus_voxceleb.bin'))

def emb(path):
    a, sr = torchaudio.load(path)
    return F.normalize(spk.forward(a[:1], sr), dim=-1)

names = sorted(n[:-4] for n in os.listdir(refs_dir) if n.endswith('.mp3'))
ref = {n: emb(os.path.join(refs_dir, n + '.mp3')) for n in names}
gen = {n: emb(os.path.join(out_dir, f'base_{n}.wav')) for n in names
       if os.path.exists(os.path.join(out_dir, f'base_{n}.wav'))}

print(f'{"generated":<14}' + ''.join(f'{n:>12}' for n in names) + '   <- reference')
for g in gen:
    row = ''.join(f'{float(gen[g] @ ref[n].T):>12.3f}' for n in names)
    print(f'{g:<14}{row}')
print('\nmatched pairs (diagonal) should clearly dominate their row/column')
