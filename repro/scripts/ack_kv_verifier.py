from pathlib import Path
root=Path('/home/chx/projects/tokenflow-repro-paper-candidate')
p=root/'python/sglang/srt/mem_cache/hiradix_cache.py';s=p.read_text()
# Preserve old diagnostics in env before invoking this script.
start=s.index('        if __import__("os").environ.get("TOKENFLOW_REPRO_SNAPSHOT_WRITES") == "1":',s.index('    def write_backup('))
end=s.index('        host_indices = self.cache_controller.write(',start)
s=s[:start]+s[end:]
s=s.replace('        self._repro_verify_kv(node, "before_evict")\n','')
old='''                check_node = end_node
                while check_node is not self.root_node:
                    self._repro_verify_kv(check_node, "after_load")
                    check_node = check_node.parent
'''
new='''                verify_nodes = getattr(self, "_repro_loaded_nodes", {}).pop(ack_id, [])
                for check_node in verify_nodes:
                    self._repro_verify_kv(check_node, "after_load")
'''
assert s.count(old)==1;s=s.replace(old,new)
old='        self.ongoing_load_back[last_hit_node.id] = last_hit_node\n'
new=old+'''        if __import__("os").environ.get("TOKENFLOW_REPRO_VERIFY_KV") == "1":
            if not hasattr(self, "_repro_loaded_nodes"):
                self._repro_loaded_nodes = {}
            self._repro_loaded_nodes[last_hit_node.id] = list(nodes_to_load)
'''
assert s.count(old)==1;s=s.replace(old,new)
needle='    def _repro_verify_kv(self, node, phase):\n'
helper='''    def _repro_verify_write_ack(self, ack_id):
        checks = getattr(self.cache_controller, "_repro_write_checks", {})
        group = checks.get(ack_id)
        if group is None:
            return
        ids, indices, saved = group
        hi = indices.long().cpu()
        host = self.cache_controller.mem_pool_host
        assert host.layout == "layer_first", host.layout
        checked = 0
        for layer, (keys, values) in enumerate(saved):
            for gpu_copy, host_buffer in ((keys, host.k_buffer), (values, host.v_buffer)):
                expected = gpu_copy.cpu()
                actual = host_buffer[layer, hi]
                assert torch.equal(actual, expected), f"Write ACK mismatch node={ack_id} layer={layer}"
                checked += expected.numel() * expected.element_size()
        logger.info("REPRO_KV write_ack_equal node=%s tokens=%d bytes=%d", ack_id, len(hi), checked)
        for node_id in ids:
            checks.pop(node_id, None)

'''
assert s.count(needle)==1;s=s.replace(needle,helper+needle)
needle='backuped_node = self.ongoing_write_through.pop(ack_id)\n'
lines=s.splitlines(True);out=[]
for line in lines:
 out.append(line)
 if needle.strip() in line:
  indent=line[:len(line)-len(line.lstrip())];out.append(indent+'self._repro_verify_write_ack(ack_id)\n')
s=''.join(out);p.write_text(s)
p=root/'python/sglang/srt/managers/cache_controller.py';s=p.read_text()
needle='''            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )'''
new='''            repro_saved = None
            if __import__("os").environ.get("TOKENFLOW_REPRO_VERIFY_KV") == "1":
                repro_saved = [(self.mem_pool_device.k_buffer[l][device_indices].clone(), self.mem_pool_device.v_buffer[l][device_indices].clone()) for l in range(self.layer_num)]
'''+needle
assert s.count(needle)==1;s=s.replace(needle,new)
needle='        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))\n'
new='''        if repro_saved is not None:
            if not hasattr(self, "_repro_write_checks"):
                self._repro_write_checks = {}
            group = (list(op.node_ids), host_indices, repro_saved)
            for node_id in op.node_ids:
                self._repro_write_checks[node_id] = group
'''+needle
assert s.count(needle)==1;s=s.replace(needle,new);p.write_text(s)
