import json
import glob
from pathlib import Path

for p in sorted(glob.glob("dataset_runs/run_*/report.json")):
    obj = json.loads(Path(p).read_text(encoding="utf-8-sig"))
    photo = obj.get("input", {}).get("photo", "")
    print(Path(p).parent.name, "->", photo, "exists=", Path(photo).exists())