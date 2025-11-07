
import os, shutil, glob, time
from pathlib import Path

ts = time.strftime("%Y%m%d_%H%M")
root = Path(f"artifacts/{ts}")
(fig_dir := root/"figures").mkdir(parents=True, exist_ok=True)
(tab_dir := root/"tables").mkdir(parents=True, exist_ok=True)

# 拷贝
for p in sorted(glob.glob("figures/*")):
    try: shutil.copy2(p, fig_dir/p.split("/")[-1])
    except Exception: pass
for p in sorted(glob.glob("tables/*")):
    try: shutil.copy2(p, tab_dir/p.split("/")[-1])
    except Exception: pass

# 清单
manifest = root/"MANIFEST.txt"
with open(manifest, "w") as f:
    f.write("# Artifacts manifest\n")
    f.write(f"path: {root}\n\n[figures]\n")
    for p in sorted(fig_dir.glob("*")):
        f.write(p.name+"\n")
    f.write("\n[tables]\n")
    for p in sorted(tab_dir.glob("*")):
        f.write(p.name+"\n")

print("[ARTIFACTS]", root)
