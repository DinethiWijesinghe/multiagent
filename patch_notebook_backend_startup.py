import json
from pathlib import Path

notebook_path = Path("d:/Multiagent/colab_clone_guide.ipynb")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))
patched = False

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "backend_proc = subprocess.Popen(" in src and "wait_http_ok(local_backend, attempts=40, delay=2, expect_status=200)" in src:
        lines = cell["source"]
        start = None
        end = None
        for i, line in enumerate(lines):
            if "backend_proc = subprocess.Popen(" in line:
                start = i
            if "wait_http_ok(local_backend, attempts=40, delay=2, expect_status=200)" in line:
                end = i
                break
        if start is None or end is None:
            raise RuntimeError("Could not locate target backend_proc block in notebook cell")

        new_block = [
            "backend_proc = subprocess.Popen(\n",
            "    [PYTHON_BIN, \"-m\", \"uvicorn\", \"multiagent.api_server:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"],\n",
            "    cwd=REPO_DIR,\n",
            "    env=backend_env,\n",
            "    stdout=subprocess.PIPE,\n",
            "    stderr=subprocess.STDOUT,\n",
            "    text=True,\n",
            ")\n",
            "print(f\"Backend PID: {backend_proc.pid}\")\n",
            "\n",
            "# Read and print initial backend logs while waiting for startup.\n",
            "start_time = time.time()\n",
            "while time.time() - start_time < 10:\n",
            "    line = backend_proc.stdout.readline()\n",
            "    if not line:\n",
            "        break\n",
            "    print(f\"[backend] {line.rstrip()}\")\n",
            "\n",
        ]

        cell["source"] = lines[:start] + new_block + lines[end:]
        patched = True
        break

if not patched:
    raise RuntimeError("No matching notebook cell was modified")

notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Notebook patched successfully")
