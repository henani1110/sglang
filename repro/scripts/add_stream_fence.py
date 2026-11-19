from pathlib import Path
root=Path('/home/chx/projects/tokenflow-repro-paper-candidate')
p=root/'python/sglang/srt/mem_cache/hiradix_cache.py';s=p.read_text();old='''        if __import__("os").environ.get("TOKENFLOW_REPRO_VERIFY_KV") == "1":
            torch.cuda.synchronize()
            device = self.cache_controller.mem_pool_device
            node._repro_written_data =''';new=old.replace('TOKENFLOW_REPRO_VERIFY_KV','TOKENFLOW_REPRO_SNAPSHOT_WRITES');assert s.count(old)==1;p.write_text(s.replace(old,new))
p=root/'python/sglang/srt/managers/scheduler.py';s=p.read_text();old='''                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )''';new=old+'''
                if __import__("os").environ.get("TOKENFLOW_REPRO_FENCE_WRITES") == "1":
                    self.tree_cache.cache_controller._repro_producer_stream = self.forward_stream
''';assert s.count(old)==1;p.write_text(s.replace(old,new))
p=root/'python/sglang/srt/managers/cache_controller.py';s=p.read_text();old='''        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
''';new=old+'''            producer = getattr(self, "_repro_producer_stream", None)
            if producer is not None:
                self.write_stream.wait_stream(producer)
''';assert s.count(old)==1;p.write_text(s.replace(old,new))
