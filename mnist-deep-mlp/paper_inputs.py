"""Offline, strict MNIST identity bridge; no network client or default cache.

Acquisition is a separate operator-reviewed action. Test images are used here
only for byte/order identity checking, never model selection or inference.
"""
import gzip
import hashlib
import json
from pathlib import Path
import struct

PROVIDER = {
    'train': ('f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf', 15561616, 60000),
    'test': ('d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b', 2595890, 10000),
}
REVISION = '77f3279092a1c1579b2250db8eafed0ad422088c'


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def idx(path, count, images):
    """Bounded canonical IDX gzip with exact header/payload/EOF."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 12000000:
        raise ValueError('Bounded ordinary IDX source required')
    size = count * (784 if images else 1)
    with gzip.open(path, 'rb') as file:
        header = file.read(16 if images else 8)
        expected = struct.pack('>IIII', 2051, count, 28, 28) if images else struct.pack('>II', 2049, count)
        if header != expected:
            raise ValueError('Canonical IDX header differs')
        payload = file.read(size + 1)
        if len(payload) != size or file.read(1):
            raise ValueError('IDX length/EOF differs')
    if not images and any(value > 9 for value in payload):
        raise ValueError('MNIST label outside0..9')
    return payload


def materialize(parquets, idx_files, output):
    """Compare every pinned provider row against independently supplied IDX."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    output = Path(output)
    output.mkdir(exist_ok=False)
    manifest = {'schemaVersion': 1, 'providerRevision': REVISION, 'license': 'MIT as declared by pinned provider card', 'splits': {}}
    for split, (expected_sha, expected_size, count) in PROVIDER.items():
        parquet = Path(parquets[split])
        if parquet.is_symlink() or parquet.stat().st_size != expected_size or digest(parquet) != expected_sha:
            raise ValueError('Pinned MNIST provider file differs')
        raw_images = idx(idx_files[split][0], count, True)
        raw_labels = idx(idx_files[split][1], count, False)
        table = pq.read_table(parquet, use_threads=False)
        if table.num_rows != count or set(table.column_names) != {'image', 'label'}:
            raise ValueError('Provider rows/schema differ')
        for i, (image, label) in enumerate(zip(table['image'].to_pylist(), table['label'].to_pylist())):
            if set(image) != {'bytes', 'path'} or type(label) is not int or label != raw_labels[i]:
                raise ValueError('Provider label/order/image schema differs')
            with Image.open(io.BytesIO(image['bytes']), formats=['PNG']) as decoded:
                if decoded.mode != 'L' or decoded.size != (28, 28) or decoded.tobytes() != raw_images[i*784:(i+1)*784]:
                    raise ValueError('Provider image/order differs from original IDX')
        records = {}
        for name, data in [('images', raw_images), ('labels', raw_labels)]:
            path = output / (split + '.' + name)
            with path.open('xb') as file:
                file.write(data)
            records[name] = {'path': path.name, 'size': len(data), 'sha256': digest(path)}
        manifest['splits'][split] = {'count': count, 'parquetSHA256': expected_sha, 'idxSHA256': [digest(p) for p in idx_files[split]], **records}
    with (output / 'MANIFEST.json').open('x') as file:
        json.dump(manifest, file, sort_keys=True)
    return manifest


def read_train(root, expected_manifest_sha256):
    """Training-only reader. Does not open or return the test split's files."""
    root = Path(root)
    path = root / 'MANIFEST.json'
    if path.is_symlink() or digest(path) != expected_manifest_sha256:
        raise ValueError('Bound dataset identity manifest differs')
    manifest = json.loads(path.read_text())
    if manifest.get('schemaVersion') != 1 or manifest.get('providerRevision') != REVISION:
        raise ValueError('Pinned dataset revision differs')
    train = manifest['splits']['train']
    if train['count'] != 60000 or train['parquetSHA256'] != PROVIDER['train'][0]:
        raise ValueError('Original training split required')
    payloads = []
    for key, size in [('images', 60000*784), ('labels', 60000)]:
        row = train[key]
        if row['path'] != 'train.' + key or row['size'] != size:
            raise ValueError('Exact training payload mapping required')
        file = root / row['path']
        if file.is_symlink() or file.stat().st_size != size or digest(file) != row['sha256']:
            raise ValueError('Training payload differs')
        payloads.append(bytearray(file.read_bytes()))
    if any(label > 9 for label in payloads[1]):
        raise ValueError('MNIST labels outside0..9')
    return payloads
