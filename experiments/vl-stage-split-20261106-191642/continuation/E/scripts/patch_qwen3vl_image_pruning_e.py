#!/usr/bin/env python3
"""Build an isolated request-scoped image keep path for the E Qwen3-VL overlay.

This is an engineering integration of fixed-K residual ranking. It is not a
claim of faithful GeoPrune reproduction: K=3, group-mean scoring, whole-group
retention, and ceil-based budget rounding are explicit adapter choices.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "d46644daaf541fc2b78992c7af0563cf1762d8089d2a55d916dded1366b3c0e8"
LOGICAL_SOURCE = "vllm/model_executor/models/qwen3_vl.py"
LOGICAL_OUTPUT = "overlay/vllm/model_executor/models/qwen3_vl.py"
LOGICAL_SNAPSHOT = "source_snapshot/qwen3_vl_before_image_pruning_e.py"
LOGICAL_DIFF = "patches/qwen3_vl_image_pruning_e_v0.5.diff"

HELPERS = r'''

def _e_get_image_keep_ratio() -> float | None:
    raw = os.environ.get("VL_IMAGE_KEEP_RATIO")
    if raw is None:
        return None
    ratio = float(raw)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("VL_IMAGE_KEEP_RATIO must be finite and in (0, 1]")
    return ratio


def _e_validate_image_keep_ratio(ratio: object) -> float:
    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("image keep ratio must be finite and in (0, 1]")
    return value


def _e_image_keep_count(
    full_token_count: int,
    keep_ratio: float | None = None,
) -> int:
    ratio = _e_get_image_keep_ratio() if keep_ratio is None else keep_ratio
    if full_token_count < 1:
        raise ValueError("full image token count must be positive")
    if ratio is None:
        return full_token_count
    ratio = _e_validate_image_keep_ratio(ratio)
    return min(full_token_count, max(1, math.ceil(full_token_count * ratio)))


def _e_resolve_request_image_keep_ratio(
    mm_processor_kwargs: Mapping[str, object],
) -> float | None:
    images_kwargs = mm_processor_kwargs.get("images_kwargs", {})
    if not isinstance(images_kwargs, Mapping):
        raise TypeError("images_kwargs must be a mapping")
    value = images_kwargs.get("e_keep_ratio", mm_processor_kwargs.get("e_keep_ratio"))
    return _e_get_image_keep_ratio() if value is None else _e_validate_image_keep_ratio(value)


def _e_strip_request_image_keep_ratio(
    mm_processor_kwargs: Mapping[str, object],
) -> dict[str, object]:
    cleaned = dict(mm_processor_kwargs)
    cleaned.pop("e_keep_ratio", None)
    images_kwargs = cleaned.get("images_kwargs")
    if isinstance(images_kwargs, Mapping):
        image_options = dict(images_kwargs)
        image_options.pop("e_keep_ratio", None)
        cleaned["images_kwargs"] = image_options
    return cleaned


def _e_normalize_image_keep_counts(
    grid_thw: torch.Tensor | list[list[int]],
    image_keep_counts: torch.Tensor | list[int] | tuple[int, ...] | None,
    merge_group_size: int,
    keep_ratio: float | None = None,
) -> list[int]:
    grid_thw_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
    full_counts = [
        int(t) * int(h) * int(w) // merge_group_size for t, h, w in grid_thw_list
    ]
    if image_keep_counts is None:
        counts = [_e_image_keep_count(full, keep_ratio) for full in full_counts]
    else:
        counts = (
            image_keep_counts.tolist()
            if isinstance(image_keep_counts, torch.Tensor)
            else list(image_keep_counts)
        )
        counts = [int(count) for count in counts]
    if len(counts) != len(full_counts):
        raise ValueError(
            f"one E keep count is required per image: got {len(counts)} counts "
            f"for {len(full_counts)} images"
        )
    for index, (count, full) in enumerate(zip(counts, full_counts)):
        if count < 1 or count > full:
            raise ValueError(
                f"image keep count at index {index} must be within [1, {full}], got {count}"
            )
    return counts


@torch.no_grad()
def _e_select_merge_group_indices(
    features: torch.Tensor,
    *,
    merge_group_size: int,
    keep_count: int,
) -> torch.Tensor:
    """Return stable merger-order indices from fixed-K residual L2 scores."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have shape [patches, hidden_size]")
    if merge_group_size < 1 or features.shape[0] % merge_group_size:
        raise ValueError("patch rows must divide evenly into merger groups")
    group_count = features.shape[0] // merge_group_size
    if keep_count < 1 or keep_count > group_count:
        raise ValueError("keep_count must be within the full merge-group count")
    if keep_count == group_count:
        return torch.arange(group_count, device=features.device)

    residual = features.to(dtype=torch.float32).clone()
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(0)
    for _ in range(3):
        v = torch.randn(
            (residual.shape[1],),
            dtype=residual.dtype,
            device=residual.device,
            generator=generator,
        )
        v = v / torch.linalg.vector_norm(v).clamp_min(torch.finfo(v.dtype).eps)
        for _ in range(10):
            u = residual @ v
            u = u / torch.linalg.vector_norm(u).clamp_min(torch.finfo(u.dtype).eps)
            v = residual.T @ u
            v = v / torch.linalg.vector_norm(v).clamp_min(torch.finfo(v.dtype).eps)
        u = residual @ v
        u = u / torch.linalg.vector_norm(u).clamp_min(torch.finfo(u.dtype).eps)
        sigma = torch.dot(u, residual @ v).abs()
        residual = residual - sigma * u[:, None] * v[None, :]

    patch_scores = torch.linalg.vector_norm(residual, ord=2, dim=1)
    group_scores = patch_scores.reshape(group_count, merge_group_size).mean(dim=1)
    ranked = torch.argsort(group_scores, descending=True, stable=True)[:keep_count]
    return torch.sort(ranked).values
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, got {count}")
    return text.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = replace_once(
        source,
        "from itertools import islice\n",
        "from itertools import islice\nimport math\nimport os\n",
        "stdlib imports",
    )
    source = replace_once(
        source,
        "import torch.nn.functional as F\n",
        "import torch.nn.functional as F\n" + HELPERS,
        "E ranker helpers",
    )
    source = replace_once(
        source,
        "        supported[\"videos_kwargs\"].update({\"min_pixels\", \"max_pixels\"})\n"
        "        return supported\n",
        "        supported[\"videos_kwargs\"].update({\"min_pixels\", \"max_pixels\"})\n"
        "        supported[\"images_kwargs\"].add(\"e_keep_ratio\")\n"
        "        return supported\n",
        "Qwen3-VL E request-scoped image budget kwarg",
    )
    source = replace_once(
        source,
        "        hashes = dict(super()._get_mm_hashes(inputs))\n",
        "        hashes = dict(super()._get_mm_hashes(inputs))\n"
        "        image_keep_ratio = _e_resolve_request_image_keep_ratio(\n"
        "            inputs.hf_processor_mm_kwargs\n"
        "        )\n"
        "        if image_keep_ratio is not None and hashes.get(\"image\"):\n"
        "            algorithm = self.info.ctx.get_mm_config().mm_hasher_algorithm\n"
        "            hashes[\"image\"] = [\n"
        "                MultiModalHasher.hash_kwargs(\n"
        "                    algorithm, mm_hash=mm_hash, e_keep_ratio=image_keep_ratio\n"
        "                )\n"
        "                for mm_hash in hashes[\"image\"]\n"
        "            ]\n",
        "request-scoped image budget cache key",
    )
    source = replace_once(
        source,
        "        merged_mm_kwargs = self.info._merge_and_resolve_mm_processor_kwargs(hf_kwargs)\n\n"
        "        if \"images_kwargs\" in merged_mm_kwargs:\n",
        "        merged_mm_kwargs = self.info._merge_and_resolve_mm_processor_kwargs(hf_kwargs)\n"
        "        image_keep_ratio = _e_resolve_request_image_keep_ratio(merged_mm_kwargs)\n\n"
        "        if \"images_kwargs\" in merged_mm_kwargs:\n",
        "request ratio resolution before HF call",
    )
    source = replace_once(
        source,
        "            image_scoped_kwargs = dict(merged_mm_kwargs[\"images_kwargs\"])\n\n"
        "            # Qwen2VLImageProcessor requires both `size` edges.",
        "            image_scoped_kwargs = dict(merged_mm_kwargs[\"images_kwargs\"])\n"
        "            image_scoped_kwargs.pop(\"e_keep_ratio\", None)\n\n"
        "            # Qwen2VLImageProcessor requires both `size` edges.",
        "strip E-only image budget before Transformers call",
    )
    source = replace_once(
        source,
        "    pruned = mm_config is not None and mm_config.is_multimodal_pruning_enabled()\n",
        "    pruned = (\n"
        "        (mm_config is not None and mm_config.is_multimodal_pruning_enabled())\n"
        "        or _e_get_image_keep_ratio() is not None\n"
        "    )\n",
        "M-RoPE pruning factor",
    )
    source = replace_once(
        source,
        "            num_tokens = int(grid_thw.prod()) // merge_length\n"
        "            return [hf_processor.image_token_id] * num_tokens\n",
        "            full_tokens = int(grid_thw.prod()) // merge_length\n"
        "            num_tokens = _e_image_keep_count(full_tokens, image_keep_ratio)\n"
        "            return [hf_processor.image_token_id] * num_tokens\n",
        "image placeholder budget",
    )
    source = replace_once(
        source,
        "        self.is_multimodal_pruning_enabled = (\n"
        "            multimodal_config.is_multimodal_pruning_enabled()\n"
        "        )\n",
        "        self.is_image_pruning_enabled = _e_get_image_keep_ratio() is not None\n"
        "        self.is_multimodal_pruning_enabled = (\n"
        "            multimodal_config.is_multimodal_pruning_enabled()\n"
        "            or self.is_image_pruning_enabled\n"
        "        )\n",
        "model image pruning flag",
    )
    source = replace_once(
        source,
        "        processed_data = self.info.ctx.call_hf_processor(\n"
        "            self.info.get_hf_processor(**non_video_mm_kwargs),\n"
        "            hf_data,\n"
        "            non_video_mm_kwargs,\n"
        "        )\n\n"
        "        # Replace each placeholder with pre-computed video tokens.\n",
        "        processed_data = self.info.ctx.call_hf_processor(\n"
        "            self.info.get_hf_processor(**non_video_mm_kwargs),\n"
        "            hf_data,\n"
        "            non_video_mm_kwargs,\n"
        "        )\n"
        "        if image_keep_ratio is not None and \"image_grid_thw\" in processed_data:\n"
        "            image_grid_thw = processed_data[\"image_grid_thw\"]\n"
        "            merge_group_size = self.info.get_hf_config().vision_config.spatial_merge_size**2\n"
        "            processed_data[\"e_image_keep_counts\"] = torch.tensor(\n"
        "                [\n"
        "                    _e_image_keep_count(\n"
        "                        int(grid.prod()) // merge_group_size, image_keep_ratio\n"
        "                    )\n"
        "                    for grid in image_grid_thw\n"
        "                ],\n"
        "                dtype=torch.int32,\n"
        "            )\n\n"
        "        # Replace each placeholder with pre-computed video tokens.\n",
        "emit per-image E keep counts",
    )
    source = replace_once(
        source,
        "        encoder_metadata: dict[str, torch.Tensor] | None = None,\n"
        "    ) -> torch.Tensor:\n",
        "        encoder_metadata: dict[str, torch.Tensor] | None = None,\n"
        "        modality: str = \"image\",\n"
        "        image_keep_counts: torch.Tensor | list[int] | tuple[int, ...] | None = None,\n"
        "    ) -> torch.Tensor:\n",
        "visual modality argument",
    )
    source = replace_once(
        source,
        "        return _create_qwen2vl_field_factory(\n"
        "            self.info.get_hf_config().vision_config.spatial_merge_size\n"
        "        )(hf_inputs)\n",
        "        fields = _create_qwen2vl_field_factory(\n"
        "            self.info.get_hf_config().vision_config.spatial_merge_size\n"
        "        )(hf_inputs)\n"
        "        return {\n"
        "            **fields,\n"
        "            \"e_image_keep_counts\": MultiModalFieldConfig.batched(\n"
        "                \"image\", keep_on_cpu=True\n"
        "            ),\n"
        "        }\n",
        "route per-image E keep counts with image modality",
    )
    source = replace_once(
        source,
        "        merge_length = image_processor.merge_size**2\n\n"
        "        def get_image_replacement_qwen3vl(item_idx: int):\n",
        "        merge_length = image_processor.merge_size**2\n"
        "        image_keep_ratio = _e_resolve_request_image_keep_ratio(\n"
        "            hf_processor_mm_kwargs\n"
        "        )\n\n"
        "        def get_image_replacement_qwen3vl(item_idx: int):\n",
        "resolve request budget for prompt expansion",
    )
    source = replace_once(
        source,
        "        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)\n"
        "        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)\n"
        "        tokenizer = self.info.get_tokenizer()\n",
        "        processor_mm_kwargs = _e_strip_request_image_keep_ratio(\n"
        "            hf_processor_mm_kwargs\n"
        "        )\n"
        "        hf_processor = self.info.get_hf_processor(**processor_mm_kwargs)\n"
        "        image_processor = self.info.get_image_processor(**processor_mm_kwargs)\n"
        "        tokenizer = self.info.get_tokenizer()\n",
        "strip E-only kwarg before HF processor construction",
    )
    source = replace_once(
        source,
        "            hidden_states = self.merger(hidden_states)\n"
        "            return torch.cat(\n"
        "                [hidden_states] + deepstack_feature_lists, dim=1\n"
        "            )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]\n",
        "            position_features = None\n"
        "            if modality == \"image\" and _e_get_image_keep_ratio() is not None:\n"
        "                if grid_thw is None:\n"
        "                    raise RuntimeError(\"image pruning requires the full image grid\")\n"
        "                grid_thw_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()\n"
        "                patch_keep_rows = []\n"
        "                merged_keep_rows = []\n"
        "                position_rows = []\n"
        "                patch_offset = 0\n"
        "                group_offset = 0\n"
        "                merge_group_size = self.spatial_merge_unit\n"
        "                all_patch_features = hidden_states.squeeze(1)\n"
        "                keep_counts = _e_normalize_image_keep_counts(\n"
        "                    grid_thw, image_keep_counts, merge_group_size\n"
        "                )\n"
        "                for size, keep_count in zip(grid_thw_list, keep_counts):\n"
        "                    t, h, w = (int(v) for v in size)\n"
        "                    if t != 1:\n"
        "                        raise RuntimeError(\"E image pruning prototype accepts image grids only\")\n"
        "                    patch_count = t * h * w\n"
        "                    if patch_count % merge_group_size:\n"
        "                        raise RuntimeError(\"image patch count is not merger aligned\")\n"
        "                    group_count = patch_count // merge_group_size\n"
        "                    image_features = all_patch_features[patch_offset : patch_offset + patch_count]\n"
        "                    selected = _e_select_merge_group_indices(\n"
        "                        image_features,\n"
        "                        merge_group_size=merge_group_size,\n"
        "                        keep_count=keep_count,\n"
        "                    )\n"
        "                    within_group = torch.arange(merge_group_size, device=selected.device)\n"
        "                    patch_rows = (selected[:, None] * merge_group_size + within_group).reshape(-1)\n"
        "                    patch_keep_rows.append(patch_rows + patch_offset)\n"
        "                    merged_keep_rows.append(selected + group_offset)\n"
        "                    original_positions = async_tensor_h2d(\n"
        "                        compute_mrope_for_media(size, self.spatial_merge_size),\n"
        "                        all_patch_features.device,\n"
        "                    )\n"
        "                    retained_positions = original_positions.index_select(0, selected)\n"
        "                    position_rows.append(\n"
        "                        torch.cat(\n"
        "                            [retained_positions, torch.zeros_like(retained_positions[:, 0:1])],\n"
        "                            dim=1,\n"
        "                        )\n"
        "                    )\n"
        "                    patch_offset += patch_count\n"
        "                    group_offset += group_count\n"
        "                patch_keep_indices = torch.cat(patch_keep_rows)\n"
        "                merged_keep_indices = torch.cat(merged_keep_rows)\n"
        "                hidden_states = hidden_states.index_select(0, patch_keep_indices)\n"
        "                deepstack_feature_lists = [\n"
        "                    feature.index_select(0, merged_keep_indices)\n"
        "                    for feature in deepstack_feature_lists\n"
        "                ]\n"
        "                position_features = torch.cat(position_rows, dim=0)\n"
        "\n"
        "            hidden_states = self.merger(hidden_states)\n"
        "            output_parts = [hidden_states] + deepstack_feature_lists\n"
        "            if position_features is not None:\n"
        "                output_parts.append(position_features.to(dtype=hidden_states.dtype))\n"
        "            return torch.cat(output_parts, dim=1)\n",
        "post-encoder image group selection",
    )
    source = replace_once(
        source,
        "        grid_thw = image_input[\"image_grid_thw\"]\n"
        "        assert grid_thw.ndim == 2\n"
        "\n"
        "        if image_input[\"type\"] == \"image_embeds\":\n",
        "        grid_thw = image_input[\"image_grid_thw\"]\n"
        "        assert grid_thw.ndim == 2\n"
        "        if self.is_image_pruning_enabled and self.use_data_parallel:\n"
        "            raise RuntimeError(\"E image pruning prototype requires the non-DP ViT path\")\n"
        "\n"
        "        if image_input[\"type\"] == \"image_embeds\":\n",
        "image pruning DP guard",
    )
    source = replace_once(
        source,
        "                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)\n",
        "                image_embeds = self.visual(\n"
        "                    pixel_values,\n"
        "                    grid_thw=grid_thw,\n"
        "                    modality=\"image\",\n"
        "                    image_keep_counts=image_input.get(\"e_image_keep_counts\"),\n"
        "                )\n",
        "direct image visual call",
    )
    source = replace_once(
        source,
        "        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()\n"
        "        return image_embeds.split(sizes)\n",
        "        if self.is_image_pruning_enabled and image_embeds.shape[-1] != self.visual.out_hidden_size + 5:\n"
        "            raise RuntimeError(\"image encoder output is missing aligned five-channel position metadata\")\n"
        "        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()\n"
        "        if self.is_image_pruning_enabled:\n"
        "            sizes = _e_normalize_image_keep_counts(\n"
        "                grid_thw,\n"
        "                image_input.get(\"e_image_keep_counts\"),\n"
        "                self.visual.spatial_merge_unit,\n"
        "            )\n"
        "        return image_embeds.split(sizes)\n",
        "image embedding split sizes",
    )
    source = replace_once(
        source,
        "        image_grid_thw = kwargs.pop(\"image_grid_thw\", None)\n\n"
        "        if pixel_values is None and image_embeds is None:\n",
        "        image_grid_thw = kwargs.pop(\"image_grid_thw\", None)\n"
        "        e_image_keep_counts = kwargs.pop(\"e_image_keep_counts\", None)\n\n"
        "        if pixel_values is None and image_embeds is None:\n",
        "parse per-image E keep-count input",
    )
    source = replace_once(
        source,
        "                pixel_values=pixel_values,\n"
        "                image_grid_thw=image_grid_thw,\n"
        "            )\n",
        "                pixel_values=pixel_values,\n"
        "                image_grid_thw=image_grid_thw,\n"
        "                e_image_keep_counts=e_image_keep_counts,\n"
        "            )\n",
        "attach E keep counts to pixel input schema",
    )
    source = replace_once(
        source,
        "                image_embeds=image_embeds,\n"
        "                image_grid_thw=image_grid_thw,\n"
        "            )\n",
        "                image_embeds=image_embeds,\n"
        "                image_grid_thw=image_grid_thw,\n"
        "                e_image_keep_counts=e_image_keep_counts,\n"
        "            )\n",
        "attach E keep counts to embedding input schema",
    )
    source = replace_once(
        source,
        "    def _process_video_input(\n"
        "        self, video_input: Qwen2_5_VLVideoInputs\n"
        "    ) -> tuple[torch.Tensor, ...]:\n"
        "        grid_thw = video_input[\"video_grid_thw\"]\n",
        "    def _process_video_input(\n"
        "        self, video_input: Qwen2_5_VLVideoInputs\n"
        "    ) -> tuple[torch.Tensor, ...]:\n"
        "        if self.is_image_pruning_enabled:\n"
        "            raise RuntimeError(\"E image pruning prototype is restricted to image inputs\")\n"
        "        grid_thw = video_input[\"video_grid_thw\"]\n",
        "image-only multimodal input guard",
    )
    source = replace_once(
        source,
        "        if self.is_multimodal_pruning_enabled:\n"
        "            merge_size = self.visual.spatial_merge_size\n"
        "            grid_thw = image_input[\"image_grid_thw\"]\n",
        "        if self.is_image_pruning_enabled:\n"
        "            return image_embeds_split\n"
        "        if self.is_multimodal_pruning_enabled:\n"
        "            merge_size = self.visual.spatial_merge_size\n"
        "            grid_thw = image_input[\"image_grid_thw\"]\n",
        "selected image position passthrough",
    )
    source = replace_once(
        source,
        "                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)\n",
        "                video_embeds = self.visual(\n"
        "                    pixel_values_videos, grid_thw=grid_thw, modality=\"video\"\n"
        "                )\n",
        "video visual call",
    )
    source = replace_once(
        source,
        "        return self.visual(pixel_values, grid_thw)\n",
        "        modality = self.get_input_modality(mm_kwargs)\n"
        "        if modality == \"video\" and self.is_image_pruning_enabled:\n"
        "            raise RuntimeError(\"E image pruning prototype is restricted to image inputs\")\n"
        "        if modality == \"image\" and self.is_image_pruning_enabled and self.use_data_parallel:\n"
        "            raise RuntimeError(\"E image pruning prototype requires the non-DP ViT path\")\n"
        "        image_keep_counts = mm_kwargs.get(\"e_image_keep_counts\")\n"
        "        return self.visual(\n"
        "            pixel_values,\n"
        "            grid_thw,\n"
        "            modality=modality,\n"
        "            image_keep_counts=image_keep_counts,\n"
        "        )\n",
        "encoder eager modality",
    )
    source = replace_once(
        source,
        "                if self.is_multimodal_pruning_enabled:\n"
        "                    video_embeddings = self._postprocess_video_embeds_evs(\n",
        "                if self.video_pruning_rate is not None:\n"
        "                    video_embeddings = self._postprocess_video_embeds_evs(\n",
        "video-only pruning condition",
    )
    source = replace_once(
        source,
        "            if self.is_multimodal_pruning_enabled:\n"
        "                assert self.video_pruning_rate is not None\n"
        "                # Compute the retention mask for each video (EVS or VidCom2).\n",
        "            if self.video_pruning_rate is not None:\n"
        "                # Compute the retention mask for each video (EVS or VidCom2).\n",
        "video retention mask condition",
    )
    source = replace_once(
        source,
        "        with gpu_sync_allowed():\n"
        "            expanded_positions[is_video_embed, :3] = original_mrope[\n"
        "                full_is_video_embed\n"
        "            ][retention_mask]\n"
        "            expanded_positions[~is_video_embed, :3] = original_mrope[\n",
        "        with gpu_sync_allowed():\n"
        "            full_video_positions = original_mrope[full_is_video_embed]\n"
        "            if retention_mask is not None:\n"
        "                full_video_positions = full_video_positions[retention_mask]\n"
        "            expanded_positions[is_video_embed, :3] = full_video_positions\n"
        "            expanded_positions[~is_video_embed, :3] = original_mrope[\n",
        "full video positions without video pruning",
    )
    source = replace_once(
        source,
        "        video_token_id: int,\n"
        "        vision_start_token_id: int,\n"
        "        vision_end_token_id: int,\n"
        "        spatial_merge_size: int,\n"
        "    ) -> Iterator[tuple[int, int, int, int]]:\n",
        "        video_token_id: int,\n"
        "        image_token_id: int,\n"
        "        vision_start_token_id: int,\n"
        "        vision_end_token_id: int,\n"
        "        spatial_merge_size: int,\n"
        "    ) -> Iterator[tuple[int, int, int, int]]:\n",
        "M-RoPE iterator image token argument",
    )
    source = replace_once(
        source,
        "                llm_grid_h = h // spatial_merge_size\n"
        "                llm_grid_w = w // spatial_merge_size\n"
        "                yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w\n",
        "                llm_grid_h = h // spatial_merge_size\n"
        "                llm_grid_w = w // spatial_merge_size\n"
        "                actual_num_tokens = 0\n"
        "                while (\n"
        "                    offset + actual_num_tokens < len(input_tokens)\n"
        "                    and input_tokens[offset + actual_num_tokens] == image_token_id\n"
        "                ):\n"
        "                    actual_num_tokens += 1\n"
        "                yield offset, llm_grid_h, llm_grid_w, actual_num_tokens\n",
        "M-RoPE actual image placeholder count",
    )
    source = replace_once(
        source,
        "            video_token_id=config.video_token_id,\n"
        "            vision_start_token_id=config.vision_start_token_id,\n",
        "            video_token_id=config.video_token_id,\n"
        "            image_token_id=config.image_token_id,\n"
        "            vision_start_token_id=config.vision_start_token_id,\n",
        "M-RoPE iterator invocation",
    )
    source = replace_once(
        source,
        "            else:\n"
        "                # Normal case: frame has exactly the expected tokens (after actual EVS\n"
        "                # pruning).\n"
        "                grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)\n"
        "                llm_pos_ids_list.append(grid_indices + text_len + st_idx)\n",
        "            else:\n"
        "                # For image keep-count placeholders, seed API-side positions with\n"
        "                # the first K grid coordinates. The worker later replaces these\n"
        "                # temporary coordinates with the selected original positions.\n"
        "                full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)\n"
        "                grid_indices = full_grid[:, :actual_num_tokens]\n"
        "                llm_pos_ids_list.append(grid_indices + text_len + st_idx)\n",
        "partial image M-RoPE positions",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = sha256(source_bytes)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected qwen3_vl.py SHA-256 {source_sha}; "
            f"expected {EXPECTED_SOURCE_SHA256}"
        )
    source = source_bytes.decode("utf-8")
    if "VL_IMAGE_KEEP_RATIO" in source or "_e_select_merge_group_indices" in source:
        raise RuntimeError("E image pruning patch already exists in source")
    patched = patch_source(source)
    patched_bytes = patched.encode("utf-8")

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot.exists():
        if sha256(args.snapshot.read_bytes()) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(f"snapshot exists with unexpected content: {args.snapshot}")
    else:
        args.snapshot.write_bytes(source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched_bytes)
    diff_text = "".join(
        difflib.unified_diff(
            source.splitlines(True),
            patched.splitlines(True),
            fromfile=f"a/{LOGICAL_SOURCE}",
            tofile=f"b/{LOGICAL_OUTPUT}",
        )
    )
    args.diff.write_text(diff_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "mode": "E-request-scoped-GeoPrune-style-image-compression",
        "source_path": LOGICAL_SOURCE,
        "source_sha256": source_sha,
        "output_path": LOGICAL_OUTPUT,
        "output_sha256": sha256(patched_bytes),
        "snapshot_path": LOGICAL_SNAPSHOT,
        "diff_path": LOGICAL_DIFF,
        "diff_sha256": sha256(diff_text.encode("utf-8")),
        "budget": "per-request images_kwargs.e_keep_ratio; fallback VL_IMAGE_KEEP_RATIO; K=max(1, ceil(full merged tokens * ratio))",
        "ranking": "fixed K=3 rank-one deflation, T=10, mean patch residual score per contiguous merger group, whole groups, stable top-k restored to merger order",
        "mrope": "selected original image coordinates are carried in the existing five-channel multimodal position envelope; API-side placeholder positions are sliced to actual K",
        "scope": "E-only; image input only; full image pixel grid still runs through the ViT; video pruning remains separate",
        "limitations": [
            "Per-request ratio transport is implemented; content-risk-aware automatic budget selection is still pending.",
            "The request budget is a ratio applied independently to each image, not a fixed total token cap across a request.",
            "Group-mean mapping and whole-group retention are engineering assumptions, not specified by GeoPrune.",
            "No answer-quality or serving-benefit result is implied by this patch.",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
