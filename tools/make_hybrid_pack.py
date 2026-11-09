#!/usr/bin/env python3
"""Build the hybrid pack: the v2 pack's dense weights and gate/up experts, the v1 pack's down (w2) experts.

The v2 pack's down is IQ2_XXS (higher quality, 32% more bytes); the v1 pack's is IQ1_S (the quant Maya's
938-1019 tok/s runs used).  Mixing them keeps v2's smaller dense weights (more VRAM expert slots) with v1's
cheaper down.  The metadata pack is symlinks + tensors.txt/sources.json/ds4.txt; the GGUFs are not copied.

  python3 tools/make_hybrid_pack.py --v1 <Maya-XS>/maya-ds4 --v2 <Maya-XS-v2>/maya-ds4-v2 \
      --out <Maya-XS-hybrid>/maya-ds4-hybrid
"""
import argparse, json, os, re, shutil

def read_tensors(pack):
    files, lines = {}, []
    for line in open(os.path.join(pack, 'tensors.txt')):
        line = line.rstrip('\n')
        if line.startswith('file '):
            _, i, name = line.split(' ', 2)
            files[int(i)] = name
        elif line.startswith('t '):
            lines.append(line)
    return files, lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', required=True, help='the v1 pack directory (contains ds4.txt)')
    ap.add_argument('--v2', required=True, help='the v2 pack directory')
    ap.add_argument('--out', required=True, help='the hybrid pack directory to write')
    a = ap.parse_args()
    v1, v2, out = os.path.abspath(a.v1), os.path.abspath(a.v2), os.path.abspath(a.out)
    model = os.path.dirname(out)   # the hybrid model directory holding the GGUF symlinks
    os.makedirs(out, exist_ok=True)
    f1, t1 = read_tensors(v1)
    f2, t2 = read_tensors(v2)
    w1 = {}
    for line in t1:
        p = line.split()
        if len(p) >= 8 and p[1].startswith('layers.') and '.ffn.experts.' in p[1] and p[1].endswith('.w2.weight'):
            w1[p[1]] = p
    used_v1 = sorted({int(w1[p[1]][3]) for line in t2 for p in [line.split()]
                      if len(p) >= 8 and p[1] in w1})
    names = [f2[i] for i in sorted(f2)] + [f1[i] for i in used_v1]
    if len(set(names)) != len(names):
        raise SystemExit('duplicate file names between the packs')
    index = {nm: i for i, nm in enumerate(names)}
    out_lines, replaced = [], 0
    for line in t2:
        p = line.split()
        if len(p) >= 8 and p[1] in w1:
            q = list(w1[p[1]])
            q[3] = str(index[f1[int(q[3])]])
            out_lines.append(' '.join(q))
            replaced += 1
        else:
            out_lines.append(line)
    with open(os.path.join(out, 'tensors.txt'), 'w') as f:
        for i, nm in enumerate(names):
            f.write(f'file {i} {nm}\n')
        for line in out_lines:
            f.write(line + '\n')
    # symlink the files the tensors reference; ../dspark and ../vision resolve from the model directory as before
    for nm in names:
        if nm.startswith('../'):
            continue
        src = os.path.join(os.path.dirname(v2), nm) if nm in f2.values() else os.path.join(os.path.dirname(v1), nm)
        dst = os.path.join(model, nm)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    # sources.json: the entries of both packs, in the hybrid file order
    by_name = {}
    for pack in (v2, v1):
        try:
            for x in json.load(open(os.path.join(pack, 'sources.json'))).get('files', []):
                by_name[x.get('name')] = x
        except (OSError, ValueError):
            pass
    json.dump({'files': [by_name[nm] for nm in names if nm in by_name]},
              open(os.path.join(out, 'sources.json'), 'w'), indent=1)
    # ds4.txt: the v2 one, its model_dir the hybrid directory
    txt = open(os.path.join(v2, 'ds4.txt')).read()
    txt = re.sub(r'(?m)^model_dir .*$', 'model_dir ' + model, txt)
    open(os.path.join(out, 'ds4.txt'), 'w').write(txt)
    for f in ('engram.bin', 'expert-profile.bin'):
        s = os.path.join(v2, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out, f))
    td = os.path.join(v2, 'tokenizer')
    if os.path.isdir(td):
        if os.path.isdir(os.path.join(out, 'tokenizer')):
            shutil.rmtree(os.path.join(out, 'tokenizer'))
        shutil.copytree(td, os.path.join(out, 'tokenizer'))
    print(f'hybrid pack {out}: {replaced} w2 tensors from v1, {len(names)} files')

if __name__ == '__main__':
    main()
