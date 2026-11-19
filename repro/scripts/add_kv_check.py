from pathlib import Path
p=Path('/home/chx/projects/tokenflow-repro-paper-candidate/python/sglang/srt/mem_cache/hiradix_cache.py');s=p.read_text()
helper='''    def _repro_verify_kv(self, node, phase):
        if __import__("os").environ.get("TOKENFLOW_REPRO_VERIFY_KV") != "1":
            return
        if node.value is None or node.host_value is None:
            return
        host = self.cache_controller.mem_pool_host
        device = self.cache_controller.mem_pool_device
        assert host.layout in ("layer_first", "page_first"), host.layout
        torch.cuda.synchronize()
        hi = node.host_value.long().cpu()
        di = node.value.long()
        checked = 0
        for layer in range(host.layer_num):
            for gpu_buf, host_buf in ((device.k_buffer[layer], host.k_buffer), (device.v_buffer[layer], host.v_buffer)):
                expected = host_buf[layer, hi] if host.layout == "layer_first" else host_buf[hi, layer]
                actual = gpu_buf[di].cpu()
                assert torch.equal(actual, expected), f"KV mismatch phase={phase} node={node.id} layer={layer}"
                checked += actual.numel() * actual.element_size()
        logger.info("REPRO_KV tensor_equal phase=%s node=%s tokens=%d bytes=%d", phase, node.id, len(di), checked)

'''
needle='    def _evict_backuped(self, node: TreeNode):\n';assert s.count(needle)==1;s=s.replace(needle,helper+needle+'        self._repro_verify_kv(node, "before_evict")\n')
needle='                end_node = self.ongoing_load_back.pop(ack_id)\n';assert s.count(needle)==1;s=s.replace(needle,needle+'                check_node = end_node\n                while check_node is not self.root_node:\n                    self._repro_verify_kv(check_node, "after_load")\n                    check_node = check_node.parent\n')
p.write_text(s)
