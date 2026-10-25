"""Recover already downloaded pip wheels for offline installation (no network)."""
from pathlib import Path
import zipfile, shutil
out = Path('/tmp/promptguard-wheels'); out.mkdir(exist_ok=True)
for p in Path.home().joinpath('.cache/pip/http-v2').rglob('*.body'):
    try:
        with p.open('rb') as f:
            if f.read(4) != b'PK\x03\x04': continue
        with zipfile.ZipFile(p) as z:
            n = next((n for n in z.namelist() if n.endswith('.dist-info/WHEEL')), None)
            if not n: continue
            tags = [x[5:] for x in z.read(n).decode().splitlines() if x.startswith('Tag: ')]
            tag = next((t for t in tags if t.startswith('py3-')), tags[0])
            dest = out / (n.split('.dist-info/')[0] + '-' + tag + '.whl')
            if not dest.exists(): shutil.copyfile(p, dest)
    except (OSError, zipfile.BadZipFile): pass
print(out)
