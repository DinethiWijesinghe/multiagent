import json
from pathlib import Path
nb = json.loads(Path('colab_clone_guide.ipynb').read_text(encoding='utf-8'))
cell = nb['cells'][14]
text = ''.join(cell['source'])
lines = text.splitlines()
print('cell lines', len(lines))
keys = [
    'Backend PID',
    'Frontend PID',
    'wait_http_ok(local_backend',
    'wait_http_ok("http://127.0.0.1:5173"',
    'start_tunnel_with_retries',
    'npm run dev',
    'print("[8/8")',
]
for i, l in enumerate(lines, start=1):
    for key in keys:
        if key in l:
            print(i, key, l)
            break
print('\n--- cell content start ---')
for i, l in enumerate(lines[:220], start=1):
    print(f'{i}: {l}')
print('--- cell content end ---')
