import json
import requests
from .logger import print_log


def load_file(fpath):
    if fpath.endswith('.jsonl'):
        with open(fpath, 'r') as fin:
            lines = fin.read().strip().split('\n')
        items = [json.loads(s) for s in lines]
    elif fpath.endswith('.txt'):
        with open(fpath, 'r') as fin:
            items = fin.read().splitlines()
    return items


def url_get(url, tries=3):
    for i in range(tries):
        if i > 0:
            print_log(f'Trying for the {i + 1} times', level='WARN')
        try:
            res = requests.get(url)
        except ConnectionError:
            continue
        if res.status_code == 200:
            return res
    print_log(f'Get {url} failed', level='WARN')
    return None
