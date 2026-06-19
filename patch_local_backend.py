import json
from pathlib import Path

nb = json.loads(Path('colab_clone_guide.ipynb').read_text(encoding='utf-8'))

patched_count = 0
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    
    # Look for wait_http_ok(local_backend pattern without definition
    if 'wait_http_ok(local_backend' in src and 'local_backend = ' not in src:
        print(f"Patching Cell {idx}...")
        lines = src.splitlines(keepends=True)
        
        new_lines = []
        for i, line in enumerate(lines):
            # Add definition right before wait_http_ok(local_backend
            if 'wait_http_ok(local_backend' in line and i > 0:
                # Check if any recent line has the definition
                found_def = False
                for j in range(max(0, i-5), i):
                    if 'local_backend =' in lines[j]:
                        found_def = True
                        break
                
                if not found_def:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + 'local_backend = "http://127.0.0.1:8000/health"\n')
                    print(f"  Added local_backend definition before wait call at line {i}")
                    patched_count += 1
            
            new_lines.append(line)
        
        cell['source'] = new_lines

# Write back
Path('colab_clone_guide.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f"✓ Notebook patched successfully ({patched_count} locations fixed)")
