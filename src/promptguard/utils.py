import hashlib
import json
import os
import platform
import random
import resource
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError: pass


def fingerprint(frame):
    cols = [c for c in ['sample_id', 'text', 'label', 'group', 'split'] if c in frame]
    return digest(frame[cols].to_json(orient='records', force_ascii=False))


def hardware():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'logical_cpus': os.cpu_count(), 'cuda': False}
    if Path('/proc/meminfo').exists():
        result['memory'] = Path('/proc/meminfo').read_text().splitlines()[:3]
    try:
        import torch
        result.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        if result['cuda']: result['gpu'] = torch.cuda.get_device_name(0)
    except ImportError: result['torch'] = 'not installed'
    return result


def size_bytes(path):
    p = Path(path)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


@contextmanager
def measured():
    start = time.perf_counter(); result = {}
    yield result
    result['wall_seconds'] = time.perf_counter() - start
    result['process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result['memory_note'] = 'Cumulative process high-water RSS on Linux; includes prior work, not isolated model memory.'
