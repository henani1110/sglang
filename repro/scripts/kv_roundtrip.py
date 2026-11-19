import json,torch,os
from pathlib import Path
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
pool=MHATokenToKVPool(size=int(os.environ.get("KV_POOL_SIZE",32)),page_size=1,dtype=torch.bfloat16,head_num=8,head_dim=128,layer_num=28,device='cuda',enable_memory_saver=False)
host=MHATokenToKVPoolHost(pool,2,int(os.environ.get("KV_HOST_GB",0)),1,'layer_first')
indices=torch.tensor([1,3,5,7,9,11,13,15],device='cuda',dtype=torch.int64)
results=[]
for backend in ['kernel','direct']:
 torch.manual_seed(42)
 for layer in range(28):
  pool.k_buffer[layer].copy_(torch.randn_like(pool.k_buffer[layer]));pool.v_buffer[layer].copy_(torch.randn_like(pool.v_buffer[layer]))
 expected=[(pool.k_buffer[l][indices].cpu(),pool.v_buffer[l][indices].cpu()) for l in range(28)]
 hi=torch.arange(8,dtype=torch.int64,device='cuda' if backend=='kernel' else 'cpu');di=indices if backend=='kernel' else indices.cpu()
 host.backup_from_device_all_layer(pool,hi,di,backend);torch.cuda.synchronize()
 bad_write=[]
 for layer in range(28):
  if not (torch.equal(host.k_buffer[layer,:8],expected[layer][0]) and torch.equal(host.v_buffer[layer,:8],expected[layer][1])):bad_write.append(layer)
 for layer in range(28):pool.k_buffer[layer][indices]=0;pool.v_buffer[layer][indices]=0
 for layer in range(28):host.load_to_device_per_layer(pool,hi,di,layer,backend)
 torch.cuda.synchronize();bad_read=[]
 for layer in range(28):
  if not (torch.equal(pool.k_buffer[layer][indices].cpu(),expected[layer][0]) and torch.equal(pool.v_buffer[layer][indices].cpu(),expected[layer][1])):bad_read.append(layer)
 results.append({'backend':backend,'jit':host.can_use_jit,'write_bad_layers':bad_write,'roundtrip_bad_layers':bad_read})
print(json.dumps(results),flush=True);Path('repro/env/kv-roundtrip.json').write_text(json.dumps(results,indent=2))
