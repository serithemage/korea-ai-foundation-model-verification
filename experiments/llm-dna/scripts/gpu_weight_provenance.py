"""Expanded Spot GPU weight-only probe: sampled AWM + selected-layer QK spectra.

No model forward pass; no FP8/MLA reconstruction. No statistical verdict threshold.
"""
import argparse
import datetime
import gc
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from weight_provenance_probe import RangeReader, unbiased_cka


def gpu_cka(x, y):
    def centered_gram(t):
        g = t @ t.T
        g.fill_diagonal_(0)
        means = g.sum(0, dtype=torch.float64) / (len(g) - 2)
        means -= means.sum() / (2 * (len(g) - 1))
        g -= means.to(g.dtype)[:, None]
        g -= means.to(g.dtype)[None, :]
        g.fill_diagonal_(0)
        return g
    a, b = centered_gram(x), centered_gram(y)
    denominator = torch.sqrt((a * a).sum() * (b * b).sum())
    if denominator <= 0:
        raise ValueError('Degenerate GPU CKA')
    return ((a * b).sum() / denominator).item()


def spectrum(q, k, heads, kv_heads):
    hd = q.shape[0] // heads
    if q.shape[0] != heads * hd or k.shape[0] != kv_heads * hd or heads % kv_heads:
        raise ValueError('Non-GQA attention geometry')
    a = q.reshape(kv_heads, heads // kv_heads, hd, q.shape[1]).sum(1).reshape(kv_heads * hd, q.shape[1])
    ra = torch.linalg.qr(a.T, mode='reduced')[1]
    rb = torch.linalg.qr(k.T, mode='reduced')[1]
    return torch.linalg.svdvals(ra @ rb.T).cpu().tolist()


def validate_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; refusing silent CPU fallback')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(32, 12)), rng.normal(size=(32, 9))
    observed = gpu_cka(torch.tensor(x, dtype=torch.float32, device='cuda'), torch.tensor(y, dtype=torch.float32, device='cuda'))
    expected = unbiased_cka(x, y)
    if abs(observed - expected) > 1e-5:
        raise RuntimeError('GPU/CPU UCKA mismatch')
    q, k = rng.normal(size=(12, 8)), rng.normal(size=(6, 8))
    repeated = np.repeat(k.reshape(2, 3, 8), 2, axis=0).reshape(12, 8)
    expected_s = np.linalg.svd(q.T @ repeated, compute_uv=False)[:6]
    observed_s = spectrum(torch.tensor(q, dtype=torch.float32, device='cuda'), torch.tensor(k, dtype=torch.float32, device='cuda'), 4, 2)
    np.testing.assert_allclose(observed_s, expected_s, rtol=1e-5, atol=1e-5)
    return {'ucka_cpu': expected, 'ucka_gpu': observed, 'gqa_identity_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared-rows', type=int, default=2048)
    args = parser.parse_args()
    began = time.monotonic()
    validation = validate_gpu()
    data = json.loads(args.evidence.read_text())
    source = {m['repo']: m for m in data['tokenizer_analysis']['models'] if 'config' in m}
    references = ['Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct']
    for repo in references:
        with urllib.request.urlopen('https://huggingface.co/api/models/' + repo, timeout=90) as response:
            meta = json.load(response)
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{meta["sha"]}/config.json', timeout=90) as response:
            config = json.load(response)
        source[repo] = {'repo': repo, 'revision': meta['sha'], 'config': config}
    selections = {
        'upstage/Solar-Open2-250B': [0, 4, 20, 44],
        'upstage/Solar-Open-100B': [0, 4, 20, 44],
        'LGAI-EXAONE/K-EXAONE-2.0-750B-A37B': [5, 25, 49, 77],
        'LGAI-EXAONE/K-EXAONE-236B-A23B': [3, 15, 31, 47],
        'skt/A.X-K2': [], 'skt/A.X-K1': [],
        references[0]: [0, 8, 16, 27], references[1]: [0, 8, 16, 27],
    }
    out = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'numpy': np.__version__,
           'validation': validation, 'models': {}, 'pairs': [],
           'scope': '2048 shared vocabulary rows per pair; 4 selected GQA layers per compatible model',
           'limits': ['Sampled AWM adaptation; not full-vocabulary/full-layer original reproduction',
                      'QK spectra only; no VO/POSA final GhostSpec score',
                      'Unrelated reference labels are experimental assumptions, not independently proven origins',
                      'No calibrated significance thresholds; no random-initialization proof',
                      'A.X MLA/FP8 attention unperformed; no model inference']}
    readers, embeddings, vocabs, excluded, attention = {}, {}, {}, {}, {}
    for repo, layers in selections.items():
        print('LOAD', repo, flush=True)
        reader = RangeReader(source[repo], args.cache)
        readers[repo] = reader
        tok = reader.small_json('tokenizer.json')
        vocabs[repo] = tok['model']['vocab']
        excluded[repo] = {t['content'] for t in tok.get('added_tokens', [])}
        embeddings[repo] = reader.tensor('model.embed_tokens.weight')
        attention[repo] = {}
        spectra = {}
        for layer in layers:
            print('ATTENTION', repo, layer, flush=True)
            prefix = f'model.layers.{layer}.self_attn.'
            q, k = reader.tensor(prefix + 'q_proj.weight'), reader.tensor(prefix + 'k_proj.weight')
            attention[repo][layer] = (q, k)
            qt, kt = torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda()
            spectra[str(layer)] = {'q_shape': list(q.shape), 'k_shape': list(k.shape),
                                   'singular_values': spectrum(qt, kt, reader.config['num_attention_heads'], reader.config['num_key_value_heads'])}
            del qt, kt
            torch.cuda.empty_cache()
        out['models'][repo] = {'revision': reader.revision, 'config': reader.config,
                               'spectra': spectra, 'range_records': reader.records}
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    pairs = [
        ('upstage/Solar-Open2-250B', 'upstage/Solar-Open-100B', 'declared self-lineage'),
        ('LGAI-EXAONE/K-EXAONE-2.0-750B-A37B', 'LGAI-EXAONE/K-EXAONE-236B-A23B', 'declared self-lineage'),
        ('skt/A.X-K2', 'skt/A.X-K1', 'target test; parent relationship not assumed'),
        (references[1], references[0], 'documented base/instruct positive control'),
        ('upstage/Solar-Open2-250B', references[0], 'unrelated-reference candidate'),
        ('LGAI-EXAONE/K-EXAONE-2.0-750B-A37B', references[0], 'unrelated-reference candidate'),
        ('skt/A.X-K2', references[0], 'unrelated-reference candidate'),
    ]
    for target, parent, role in pairs:
        print('COMPARE', target, parent, flush=True)
        common = sorted((set(vocabs[target]) & set(vocabs[parent])) - excluded[target] - excluded[parent])
        if len(common) < args.shared_rows:
            raise ValueError('Insufficient shared vocabulary')
        positions = np.linspace(0, len(common) - 1, args.shared_rows, dtype=int)
        tokens = [common[i] for i in positions]
        ti, pi = [vocabs[target][t] for t in tokens], [vocabs[parent][t] for t in tokens]
        et, ep = embeddings[target][ti], embeddings[parent][pi]
        train, heldout = np.arange(0, len(tokens), 2), np.arange(1, len(tokens), 2)
        a, b = torch.from_numpy(ep[train]).cuda(), torch.from_numpy(et[train]).cuda()
        a = torch.nn.functional.normalize(a.T, dim=1)
        b = torch.nn.functional.normalize(b.T, dim=1)
        similarity = (a @ b.T).cpu().numpy()
        rows, cols = linear_sum_assignment(-np.abs(similarity))
        signs = np.where(similarity[rows, cols] >= 0, 1.0, -1.0).astype('float32')
        def mean_cos(x, y):
            norms = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
            return float(np.mean(np.sum(x * y, axis=0) / np.maximum(norms, 1e-12)))
        entry = {'target': target, 'parent': parent, 'role': role, 'shared_token_count': len(common),
                 'sample_tokens': tokens, 'target_token_ids': ti, 'parent_token_ids': pi,
                 'train_rows': len(train), 'heldout_rows': len(heldout),
                 'training_column_cosine': mean_cos(ep[train][:, rows] * signs, et[train][:, cols]),
                 'heldout_column_cosine': mean_cos(ep[heldout][:, rows] * signs, et[heldout][:, cols]),
                 'parent_columns': rows.tolist(), 'target_columns': cols.tolist(), 'signs': signs.tolist(),
                 'attention_ucka': []}
        del a, b, similarity
        for pl, (pq, pk) in attention[parent].items():
            pa = [torch.from_numpy(w[:, rows] * signs).T.cuda() for w in [pq, pk]]
            for tl, (tq, tk) in attention[target].items():
                ta = [torch.from_numpy(w[:, cols]).T.cuda() for w in [tq, tk]]
                qscore, kscore = gpu_cka(pa[0], ta[0]), gpu_cka(pa[1], ta[1])
                if not np.isfinite([qscore, kscore]).all():
                    raise ValueError('Nonfinite UCKA score')
                entry['attention_ucka'].append({'parent_layer': pl, 'target_layer': tl, 'q': qscore, 'k': kscore})
                del ta
            del pa
        out['pairs'].append(entry)
        out['elapsed_seconds'] = time.monotonic() - began
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print('RESULT', target, parent, entry['heldout_column_cosine'], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    out['complete'] = True
    out['elapsed_seconds'] = time.monotonic() - began
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print('COMPLETE', out['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
