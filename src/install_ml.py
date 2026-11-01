"""Fetch the tested official YOLOX model and validate its pinned SHA256."""
import hashlib
import urllib.request
from object_detector import DEFAULT_MODEL, MODEL_SHA256, MODEL_URL


def main():
    target = DEFAULT_MODEL
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == MODEL_SHA256:
        print('Local AI model checksum verified.')
        return
    with urllib.request.urlopen(MODEL_URL, timeout=180) as response:
        blob = response.read(25_000_000)
    if hashlib.sha256(blob).hexdigest() != MODEL_SHA256:
        raise SystemExit('Model checksum mismatch; refusing to install.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    temp.write_bytes(blob)
    temp.replace(target)
    print('Official YOLOX tiny model installed; tested artifact checksum verified.')


if __name__ == '__main__': main()
