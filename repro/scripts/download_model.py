import json,time
from pathlib import Path
from huggingface_hub import snapshot_download
revision="f2826a00ceef68f0f2b946d945ecc0477ce4450c"
path=Path("repro/models/Qwen2-7B-Instruct")
start=time.time(); result=snapshot_download("Qwen/Qwen2-7B-Instruct",revision=revision,local_dir=str(path),allow_patterns=["*.json","*.safetensors","*.txt","*.model","LICENSE","README.md"],max_workers=4)
record={"repo_id":"Qwen/Qwen2-7B-Instruct","revision":revision,"path":str(path.resolve()),"download_s":time.time()-start}
Path("repro/env/model-7b.json").write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
