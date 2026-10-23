from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ORDER = ["fig1_2", "fig3_4_5", "fig6", "fig7_8_9", "fig10_11", "fig12_13"]
for name in ORDER:
    print(f"Running {name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py")], cwd=ROOT, check=True)
