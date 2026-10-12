"""Limited CPU provenance probe; not a complete AWM/GhostSpec reproduction.

Uses 512 embedding rows for LAP alignment and one selected GQA layer per model.
No forward inference, no model code execution, no AWS resources. See docs/09.
"""
import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.linalg import qr, svdvals
from scipy.optimize import linear_sum_assignment


def align_columns(parent, child):
    """AWM-style signed LAP using a training subset of shared embedding rows."""
    a = parent / np.maximum(np.linalg.norm(parent, axis=0), 1e-12)
    b = child / np.maximum(np.linalg.norm(child, axis=0), 1e-12)
    similarity = a.T @ b
    rows, cols = linear_sum_assignment(-np.abs(similarity))
    signs = np.where(similarity[rows, cols] >= 0, 1.0, -1.0)
    return rows, cols, signs


def unbiased_cka(x, y):
    """Linear U-centered CKA, following the AWM reference equations."""
    if len(x) != len(y) or len(x) < 4:
        raise ValueError('UCKA requires >=4 corresponding observations')
    def gram(values):
        g = values @ values.T
        np.fill_diagonal(g, 0)
        means = g.sum(axis=0, dtype=np.float64) / (len(g) - 2)
        means -= means.sum() / (2 * (len(g) - 1))
        g = g - means[:, None] - means[None, :]
        np.fill_diagonal(g, 0)
        return g
    a, b = gram(x), gram(y)
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denominator <= 0:
        raise ValueError('Degenerate UCKA Gram matrix')
    return float(np.sum(a * b) / denominator)


def gqa_product_spectrum(q, k, query_heads, kv_heads, head_dim):
    """Exact nonzero spectrum of Q.T @ repeat_K without the dense product.

Summing the Q heads sharing each K gives A; thin QR of A.T and K.T means
A.T @ K has the same nonzero singular values as R_A @ R_K.T.
"""
    hidden = q.shape[1]
    if query_heads % kv_heads or q.shape != (query_heads * head_dim, hidden):
        raise ValueError('Incompatible query geometry')
    if k.shape != (kv_heads * head_dim, hidden):
        raise ValueError('Incompatible key geometry')
    summed = q.reshape(kv_heads, query_heads // kv_heads, head_dim, hidden).sum(axis=1)
    _, ra = qr(summed.reshape(kv_heads * head_dim, hidden).T, mode='economic')
    _, rb = qr(k.T, mode='economic')
    return svdvals(ra @ rb.T)


class RangeReader:
    def __init__(self, model, cache):
        self.repo, self.revision, self.config = model['repo'], model['revision'], model['config']
        self.cache = cache / self.repo.replace('/', '__') / self.revision
        self.cache.mkdir(parents=True, exist_ok=True)
        self.headers, self.records = {}, []
        self.index = self.small_json('model.safetensors.index.json')['weight_map']

    def url(self, filename):
        return f'https://huggingface.co/{self.repo}/resolve/{self.revision}/{filename}'

    def small_json(self, filename):
        path = self.cache / filename
        if not path.exists():
            with urllib.request.urlopen(self.url(filename), timeout=90) as response:
                path.write_bytes(response.read())
        return json.loads(path.read_text())

    def ranged(self, filename, start, size):
        path = self.cache / f'{filename}.{start}-{start + size - 1}.bin'
        if path.exists():
            data = path.read_bytes()
        else:
            url = self.url(filename) + f'?download=true&probe_range={start}-{start + size - 1}'
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + size - 1}'})
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = f'bytes {start}-{start + size - 1}'
                if response.status != 206 or response.headers.get('Content-Range', '').split('/')[0] != expected:
                    raise ValueError('Server did not honor exact byte range')
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError('Invalid range length')
            path.write_bytes(data)
        if len(data) != size:
            raise ValueError('Invalid cached range length')
        self.records.append({'file': filename, 'start': start, 'bytes': size,
                             'sha256': hashlib.sha256(data).hexdigest()})
        return data

    def tensor(self, name, first_row=0, rows=None):
        filename = self.index[name]
        if filename not in self.headers:
            length = int.from_bytes(self.ranged(filename, 0, 8), 'little')
            if length > 20_000_000:
                raise ValueError('Unexpected header length')
            self.headers[filename] = length, json.loads(self.ranged(filename, 8, length))
        length, header = self.headers[filename]
        item = header[name]
        if item['dtype'] != 'BF16' or len(item['shape']) != 2:
            raise ValueError(f'{name}: unsupported tensor {item["dtype"]} {item["shape"]}')
        total_rows, hidden = item['shape']
        rows = total_rows if rows is None else rows
        if first_row < 0 or first_row + rows > total_rows:
            raise ValueError('Rows outside tensor')
        start = 8 + length + item['data_offsets'][0] + first_row * hidden * 2
        raw = self.ranged(filename, start, rows * hidden * 2)
        values = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite BF16 values')
        return values.reshape(rows, hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text())
    models = {m['repo']: m for m in data['tokenizer_analysis']['models'] if 'config' in m}
    pairs = [
        ('upstage/Solar-Open2-250B', 'upstage/Solar-Open-100B', 0, 0),
        ('LGAI-EXAONE/K-EXAONE-2.0-750B-A37B', 'LGAI-EXAONE/K-EXAONE-236B-A23B', 5, 3),
        ('skt/A.X-K2', 'skt/A.X-K1', None, None),
    ]
    output = {'scope': 'AWM-inspired sampled embedding alignment + single-layer GQA spectral pilot',
              'aws_resources_created': False, 'numpy': np.__version__, 'pairs': [],
              'limits': ['Not full AWM/GhostSpec reproduction', 'No independent real-model negative controls',
                         'No calibrated significance threshold', 'Not proof of random initialization',
                         'No POSA/full-layer matching', 'No FP8/MLA adapter']}
    for target_repo, parent_repo, target_layer, parent_layer in pairs:
        began = time.monotonic()
        print('PAIR', target_repo, flush=True)
        target, parent = RangeReader(models[target_repo], args.cache), RangeReader(models[parent_repo], args.cache)
        vt = target.small_json('tokenizer.json')['model']['vocab']
        vp = parent.small_json('tokenizer.json')['model']['vocab']
        it, ip = {v: k for k, v in vt.items()}, {v: k for k, v in vp.items()}
        row_ids = [i for start in [4096, 32768, 65536, 98304] for i in range(start, start + 128)]
        if any(it[i] != ip[i] for i in row_ids):
            raise ValueError('Sampled token IDs are not semantically aligned')
        et = np.concatenate([target.tensor('model.embed_tokens.weight', s, 128) for s in [4096, 32768, 65536, 98304]])
        ep = np.concatenate([parent.tensor('model.embed_tokens.weight', s, 128) for s in [4096, 32768, 65536, 98304]])
        train, heldout = np.arange(0, 512, 2), np.arange(1, 512, 2)
        r, c, signs = align_columns(ep[train].astype('float64'), et[train].astype('float64'))
        def column_cos(a, b):
            norms = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
            return float(np.mean(np.sum(a * b, axis=0) / np.maximum(norms, 1e-12)))
        result = {'target': target_repo, 'parent': parent_repo,
                  'target_revision': target.revision, 'parent_revision': parent.revision,
                  'embedding_token_ids': row_ids, 'train_rows': 256, 'heldout_rows': 256,
                  'aligned_training_column_cosine': column_cos(ep[train][:, r] * signs, et[train][:, c]),
                  'heldout_column_cosine_before': column_cos(ep[heldout], et[heldout]),
                  'heldout_column_cosine_after': column_cos(ep[heldout][:, r] * signs, et[heldout][:, c]),
                  'identity_assignment_fraction': float(np.mean(r == c)),
                  'assignment_parent': r.tolist(), 'assignment_target': c.tolist(), 'assignment_sign': signs.tolist()}
        if target_layer is not None:
            spectra, weights = {}, {}
            for label, reader, layer in [('target', target, target_layer), ('parent', parent, parent_layer)]:
                prefix = f'model.layers.{layer}.self_attn.'
                q, k = reader.tensor(prefix + 'q_proj.weight'), reader.tensor(prefix + 'k_proj.weight')
                cfg = reader.config
                hd = cfg['head_dim']
                s = gqa_product_spectrum(q, k, cfg['num_attention_heads'], cfg['num_key_value_heads'], hd)
                spectra[label] = {'layer': layer, 'head_dim': hd, 'q_shape': list(q.shape),
                                  'k_shape': list(k.shape), 'singular_values': s.tolist()}
                weights[label] = (q, k)
            result['spectra'] = spectra
            result['attention_ucka'] = {}
            for name, index in [('q', 0), ('k', 1)]:
                a, b = weights['parent'][index], weights['target'][index]
                result['attention_ucka'][name] = unbiased_cka((a[:, r] * signs).T, b[:, c].T)
            a, b = np.asarray(spectra['target']['singular_values']), np.asarray(spectra['parent']['singular_values'])
            result['spectrum_cosine'] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            result['spectrum_relative_l2'] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            del weights
        else:
            result['attention_status'] = 'MLA/SGA + FP8 adapter not implemented; no attention score'
        result['range_records'] = {'target': target.records, 'parent': parent.records}
        result['elapsed_seconds'] = time.monotonic() - began
        output['pairs'].append(result)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ['spectra', 'range_records', 'embedding_token_ids', 'assignment_parent', 'assignment_target', 'assignment_sign']}), flush=True)
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
