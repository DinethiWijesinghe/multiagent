import json
from pathlib import Path

nb = json.loads(Path('colab_clone_guide.ipynb').read_text(encoding='utf-8'))
print(f'Total cells: {len(nb["cells"])}')
print('\nCells with wait_http_ok(local_backend:')

all_fixed = True
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'wait_http_ok(local_backend' in src:
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if 'wait_http_ok(local_backend' in line:
                # Check if defined in previous lines
                found_def = False
                for j in range(max(0, i-15), i):
                    if 'local_backend =' in lines[j]:
                        found_def = True
                        break
                
                status = '✓ FIXED' if found_def else '✗ NEEDS FIX'
                print(f'  {status} - Cell {idx}: line {i+1}')
                if found_def:
                    print(f'      Definition found at line {j+1}: {lines[j][:60]}')
                else:
                    all_fixed = False

print(f'\n{"="*50}')
if all_fixed:
    print('✓ SUCCESS: All wait_http_ok(local_backend calls are fixed!')
else:
    print('✗ ISSUE: Some calls still missing local_backend definition')
