import json, traceback, glob
from core.report_html import generate_html

files = sorted(glob.glob("output/*_report_*.json"))
path = files[-1]
print("Using:", path)
d = json.load(open(path, encoding="utf-8"))
print("cloud_annotations arg: {} (explicit empty)")
try:
    res = generate_html(d["video"], d["moments"], report_data=d, cloud_annotations={})
    print("OK ->", res)
except Exception:
    traceback.print_exc()

# Проверим структуру moments - может class это dict
print("\n--- moment[0] objects sample ---")
m0 = d["moments"][0] if d.get("moments") else {}
for o in m0.get("objects", [])[:3]:
    print(repr(o))