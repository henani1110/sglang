#!/usr/bin/env python3
"""Instrument the vLLM 0.33 V2 encoder path used by the pilot server."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "artifacts" / "original_files"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {count}")
    backup = BACKUP_DIR / path.name
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new, 1))


def package_dir() -> Path:
    out = subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1]
    return Path(out)


def main() -> None:
    package = package_dir()
    encoder_runner = package / "v1" / "worker" / "gpu" / "mm" / "encoder_runner.py"
    replace_once(
        encoder_runner,
        "import torch\n\nfrom vllm.logger import init_logger\n",
        "import torch\n\nfrom vllm.logger import init_logger\n"
        "from vllm.vl_pilot_hook import mm_keys_context\n",
    )
    replace_once(
        encoder_runner,
        """    def execute_mm_encoder(
        self, mm_kwargs: list[tuple[str, MultiModalKwargsItem]]
    ) -> list[torch.Tensor]:
        # Stable-sort items by modality so each modality is encoded in as few
        # batches as possible. The returned outputs are restored to input order.
        order = sorted(range(len(mm_kwargs)), key=lambda i: mm_kwargs[i][0])
        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            [mm_kwargs[i] for i in order], device=self.device, pin_memory=PIN_MEMORY
        ):
            cg_manager = self.cudagraph_manager
            cudagraph_output = (
                cg_manager.execute(mm_kwargs_batch)
                if cg_manager is not None
                and cg_manager.is_captured()
                and cg_manager.supports_modality(modality)
                else None
            )
            batch_outputs = (
                cudagraph_output
                if cudagraph_output is not None
                else self.model.embed_multimodal(**mm_kwargs_batch)
            )
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)
        return [encoder_outputs[i] for i in np.argsort(order)]
""",
        """    def execute_mm_encoder(
        self,
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
        mm_hashes: list[str] | None = None,
    ) -> list[torch.Tensor]:
        # Stable-sort items by modality so each modality is encoded in as few
        # batches as possible. The returned outputs are restored to input order.
        order = sorted(range(len(mm_kwargs)), key=lambda i: mm_kwargs[i][0])
        encoder_outputs: list[torch.Tensor] = []
        current_item_idx = 0
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            [mm_kwargs[i] for i in order], device=self.device, pin_memory=PIN_MEMORY
        ):
            batch_keys = (
                [mm_hashes[i] for i in order[current_item_idx : current_item_idx + num_items]]
                if mm_hashes is not None
                else []
            )
            with mm_keys_context(batch_keys):
                cg_manager = self.cudagraph_manager
                cudagraph_output = (
                    cg_manager.execute(mm_kwargs_batch)
                    if cg_manager is not None
                    and cg_manager.is_captured()
                    and cg_manager.supports_modality(modality)
                    else None
                )
                batch_outputs = (
                    cudagraph_output
                    if cudagraph_output is not None
                    else self.model.embed_multimodal(**mm_kwargs_batch)
                )
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)
            current_item_idx += num_items
        return [encoder_outputs[i] for i in np.argsort(order)]
""",
    )

    model_state = package / "v1" / "worker" / "gpu" / "model_states" / "interface.py"
    replace_once(
        model_state,
        "encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)",
        "encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs, mm_hashes)",
    )

    runner = package / "v1" / "worker" / "gpu" / "model_runner.py"
    replace_once(
        runner,
        "from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput\n",
        "from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput\n"
        "from vllm.vl_pilot_hook import (\n"
        "    flush_cuda_events,\n"
        "    flush_scheduler_events,\n"
        "    record_scheduler_event,\n"
        ")\n",
    )
    replace_once(
        runner,
        """    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            self.pooling_runner.clear()

    def release_late_interaction_query_cache(self, query_keys: list[str]) -> None:
""",
        """    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            self.pooling_runner.clear()

    def pilot_flush_visual_events(self) -> dict[str, object]:
        return {
            "cuda_events": flush_cuda_events(),
            "scheduler_flushed": flush_scheduler_events(),
        }

    def release_late_interaction_query_cache(self, query_keys: list[str]) -> None:
""",
    )
    replace_once(
        runner,
        """        if not dummy_run:
            # Update the request states.
""",
        """        if not dummy_run:
            record_scheduler_event(scheduler_output)
            # Update the request states.
""",
    )

    worker = package / "v1" / "worker" / "gpu_worker.py"
    replace_once(
        worker,
        "    @torch.inference_mode()\n    @with_gpu_sync_check\n    def execute_model(\n",
        "    def pilot_flush_visual_events(self) -> dict[str, object]:\n"
        "        return self.model_runner.pilot_flush_visual_events()\n\n"
        "    @torch.inference_mode()\n    @with_gpu_sync_check\n    def execute_model(\n",
    )

    print(f"patched_v2_encoder_runner={encoder_runner}")
    print(f"patched_v2_model_state={model_state}")
    print(f"patched_v2_runner={runner}")
    print(f"patched_gpu_worker={worker}")
    print(f"backup_dir={BACKUP_DIR}")


if __name__ == "__main__":
    main()
