from pathlib import Path
p=Path('/home/chx/projects/tokenflow-repro-paper-candidate/python/sglang/srt/mem_cache/hiradix_cache.py');s=p.read_text()
old='    def write_backup(self, node: TreeNode, write_back=False):\n'
new=old+'''        if __import__("os").environ.get("TOKENFLOW_REPRO_VERIFY_KV") == "1":
            torch.cuda.synchronize()
            device = self.cache_controller.mem_pool_device
            node._repro_written_data = [(device.k_buffer[l][node.value].cpu(), device.v_buffer[l][node.value].cpu()) for l in range(device.layer_num)]
'''
assert s.count(old)==1;s=s.replace(old,new)
old='                    raise AssertionError(f"KV mismatch phase={phase} node={node.id} layer={layer}")'
new='''                    saved = getattr(node, "_repro_written_data", None)
                    if saved is not None:
                        logger.error("REPRO_KV snapshot_compare node=%s layer=%s host_equals_saved_k=%s device_equals_saved_k=%s device_indices=%s host_indices=%s", node.id, layer, torch.equal(host.k_buffer[layer, hi], saved[layer][0]), torch.equal(device.k_buffer[layer][di].cpu(), saved[layer][0]), di.cpu().tolist(), hi.tolist())
                    raise AssertionError(f"KV mismatch phase={phase} node={node.id} layer={layer}")'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
