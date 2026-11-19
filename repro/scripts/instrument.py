from pathlib import Path
root=Path('/home/chx/projects/tokenflow-repro-historical')
def patch(file, old, new):
    p=root/file; s=p.read_text(); assert s.count(old)==1,(file,s.count(old)); p.write_text(s.replace(old,new))
p='python/sglang/srt/mem_cache/hiradix_cache.py'
patch(p,'        return len(host_indices)\n','        if __import__("os").environ.get("TOKENFLOW_REPRO_TRACE") == "1":\n            logger.info("REPRO_KV write_queued node=%s tokens=%d", node.id, len(host_indices))\n        return len(host_indices)\n')
patch(p,'        return device_indices\n\n    def init_load_back(', '        if __import__("os").environ.get("TOKENFLOW_REPRO_TRACE") == "1":\n            logger.info("REPRO_KV load_queued node=%s tokens=%d", last_hit_node.id, len(device_indices))\n        return device_indices\n\n    def init_load_back(')
patch(p,'        num_evicted = self.cache_controller.evict_device(node.value)\n','        if __import__("os").environ.get("TOKENFLOW_REPRO_TRACE") == "1":\n            logger.info("REPRO_KV evict_backed node=%s tokens=%d", node.id, len(node.value))\n        num_evicted = self.cache_controller.evict_device(node.value)\n')
patch(p,'                end_node = self.ongoing_load_back.pop(ack_id)\n','                end_node = self.ongoing_load_back.pop(ack_id)\n                if __import__("os").environ.get("TOKENFLOW_REPRO_TRACE") == "1":\n                    logger.info("REPRO_KV load_done node=%s", ack_id)\n')
p='python/sglang/srt/managers/schedule_policy.py'
patch(p,'                release_counter += 1\n','                release_counter += 1\n                if os.environ.get("TOKENFLOW_REPRO_TRACE") == "1":\n                    logger.info("REPRO_KV preempt rid=%s output_tokens=%d", running_req.rid, len(running_req.output_ids))\n')
