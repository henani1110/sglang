#!/usr/bin/env python3
"""Apply the isolated vLLM visual encoder hook used by this pilot."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "artifacts" / "original_files"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {count}")
    if not (BACKUP_DIR / path.name).exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, BACKUP_DIR / path.name)
    path.write_text(source.replace(old, new, 1))


def main() -> None:
    out = subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1]
    package = Path(out)
    if not (package / "v1" / "worker" / "gpu_model_runner.py").is_file():
        raise RuntimeError(f"unexpected vLLM package directory: {package}")

    hook_path = package / "vl_pilot_hook.py"
    if not hook_path.exists():
        shutil.copy2(ROOT / "scripts" / "vl_pilot_hook.py", hook_path)
    runner = package / "v1" / "worker" / "gpu_model_runner.py"
    replace_once(
        runner,
        "import vllm.envs as envs\n",
        "import vllm.envs as envs\nfrom vllm.vl_pilot_hook import flush_cuda_events, mm_keys_context\n",
    )
    replace_once(
        runner,
        """                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )
""",
        """                        with mm_keys_context(
                            mm_hashes[current_item_idx + video_idx : current_item_idx + video_idx + 1]
                        ):
                            micro_batch_outputs = model.embed_multimodal(
                                **micro_batch_mm_inputs
                            )
""",
    )
    replace_once(
        runner,
        """                        batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
""",
        """                        with mm_keys_context(
                            mm_hashes[current_item_idx : current_item_idx + num_items]
                        ):
                            batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
""",
    )
    replace_once(
        runner,
        """        self.encoder_cache.clear()
        self.late_interaction_runner.clear()

    def release_late_interaction_query_cache(self, query_keys: list[str]) -> None:
""",
        """        self.encoder_cache.clear()
        self.late_interaction_runner.clear()

    def pilot_flush_visual_events(self) -> list[dict[str, object]]:
        return flush_cuda_events()

    def release_late_interaction_query_cache(self, query_keys: list[str]) -> None:
""",
    )

    model_path = package / "model_executor" / "models" / "qwen3_vl.py"
    old_forward = """    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states = self.input_norm(
            x.to(device=self.device, non_blocking=True), self.dtype
        )
        hidden_states = self.patch_embed(hidden_states)

        if encoder_metadata is None:
            if isinstance(grid_thw, list):
                grid_thw_list = grid_thw
            else:
                grid_thw_list = grid_thw.tolist()
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

        pos_embeds = encoder_metadata["pos_embeds"]
        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
                rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
                max_seqlen=encoder_metadata["max_seqlen"],
                sequence_lengths=encoder_metadata.get("sequence_lengths"),
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states
"""
    new_forward = """    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        from vllm.vl_pilot_hook import pilot_visual_forward

        pixel_values = x.to(device=self.device, non_blocking=True)

        def compute_visual_features() -> torch.Tensor:
            hidden_states = self.input_norm(pixel_values, self.dtype)
            hidden_states = self.patch_embed(hidden_states)

            local_metadata = encoder_metadata
            if local_metadata is None:
                if isinstance(grid_thw, list):
                    grid_thw_list = grid_thw
                else:
                    grid_thw_list = grid_thw.tolist()
                local_metadata = self.prepare_encoder_metadata(grid_thw_list)

            pos_embeds = local_metadata["pos_embeds"]
            hidden_states = hidden_states + pos_embeds
            hidden_states = hidden_states.unsqueeze(1)

            deepstack_feature_lists = []
            for layer_num, blk in enumerate(self.blocks):
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=local_metadata["cu_seqlens"],
                    rotary_pos_emb_cos=local_metadata["rotary_pos_emb_cos"],
                    rotary_pos_emb_sin=local_metadata["rotary_pos_emb_sin"],
                    max_seqlen=local_metadata["max_seqlen"],
                    sequence_lengths=local_metadata.get("sequence_lengths"),
                )
                if layer_num in self.deepstack_visual_indexes:
                    deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                    deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                        hidden_states
                    )
                    deepstack_feature_lists.append(deepstack_feature)
            hidden_states = self.merger(hidden_states)
            return torch.cat(
                [hidden_states] + deepstack_feature_lists, dim=1
            )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]

        return pilot_visual_forward(self, pixel_values, grid_thw, compute_visual_features)
"""
    replace_once(model_path, old_forward, new_forward)

    print(f"patched_package={package}")
    print(f"hook={hook_path}")
    print(f"backup_dir={BACKUP_DIR}")


if __name__ == "__main__":
    main()
