"""Download a pinned OmniRetarget terrain subset locally, then transfer to server.

Only downloads and verifies files; never executes upstream code or runs physics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request

REPO='omniretarget/OmniRetarget_Dataset'
REVISION='135259130158f1cb1799e4ab9b6566c938285fc9'


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'supermimic-source-audit/1'}),timeout=120)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision):p.error('An immutable 40-character revision is required')
    if not 1<=a.workers<=12:p.error('workers must be 1..12')
    api=f'https://huggingface.co/api/datasets/{REPO}/tree/{a.revision}?recursive=true&limit=1000'
    with request(api) as response:
        entries=json.load(response)
        if 'rel="next"' in response.headers.get('Link',''):raise RuntimeError('Index is paginated; refuse an incomplete manifest')
    selected=[x for x in entries if x['type']=='file' and '.DS_Store' not in x['path'] and
              (x['path'] in ('README.md','visualize.py','robot-terrain.zip','models/ground_box.sdf') or
               x['path'].startswith(('models/g1/','models/terrain/')))]
    a.out.mkdir(parents=True,exist_ok=True)

    def fetch(entry):
        name=entry['path'];relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe remote path')
        dest=a.out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{a.revision}/'+urllib.parse.quote(name)
        expected=entry.get('lfs',{}).get('oid')
        def valid(data):
            if len(data)!=entry['size']:return False
            if expected:return hashlib.sha256(data).hexdigest()==expected
            return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['oid']
        if dest.exists():
            data=dest.read_bytes()
            if not valid(data):raise ValueError(f'Existing file differs: {name}; preserve it and use another output directory')
        else:
            for attempt in range(3):
                try:
                    with request(url) as response:data=response.read()
                    if not valid(data):raise ValueError(f'Source checksum mismatch: {name}')
                    temporary=dest.with_name(dest.name+'.download')
                    with temporary.open('xb') as output:output.write(data)
                    temporary.rename(dest)
                    break
                except (urllib.error.URLError,TimeoutError):
                    if attempt==2:raise
                    time.sleep(2**attempt)
        return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':url}

    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(fetch,selected):
            rows.append(row)
            if len(rows)%40==0:print(f'verified={len(rows)}/{len(selected)}',flush=True)
    manifest={'repository':REPO,'revision':a.revision,'license_from_dataset_card':'MIT','files':rows,
              'total_bytes':sum(r['bytes'] for r in rows),'scope':'Source terrain motions and original model assets; no X2 conversion or feasibility claim'}
    (a.out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'download-as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':main()
