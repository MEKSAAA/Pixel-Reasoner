import torch
from vllm.worker.worker import Worker

from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging_utils import init_logger
from .utils import get_physical_gpu_id
import re
from difflib import get_close_matches


logger = init_logger(__name__)


class WorkerWrap(Worker):
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
        self._model_update_with_ray = use_ray
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    # === 新增：去掉常见训练侧前缀 ===
    def _strip_prefix(self, name: str) -> str:
        for p in ("model.", "module."):
            if name.startswith(p):
                return name[len(p):]
        return name

    # === 新增：HF→vLLM 名字映射（含启发式 + 显式表）===
    def _map_name_hf_to_vllm(self, name: str) -> str:
        # 0) 先去前缀
        name = self._strip_prefix(name)

        # 1) 若 vLLM 模型自带 mapper，优先调用
        mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)
        if callable(mapper):
            name = mapper(name)

        # 2) 显式一对一映射（把你确认过的键补进来）
        explicit = {
            # 例：HF:  visual.patch_embed.proj.weight
            #     vLLM: vision_tower.vision_model.embeddings.patch_embed.proj.weight
            "visual.patch_embed.proj.weight": "vision_tower.vision_model.embeddings.patch_embed.proj.weight",
            "visual.patch_embed.proj.bias":   "vision_tower.vision_model.embeddings.patch_embed.proj.bias",
            # TODO: 后面你打印到具体键名后，在这里继续补充
        }
        if name in explicit:
            return explicit[name]

        # 3) 启发式规则（根据常见 Qwen2.5-VL 命名差异）
        # visual.* → vision_tower.vision_model.*
        if name.startswith("visual."):
            name = name.replace("visual.", "vision_tower.vision_model.", 1)

        # blocks.N. → encoder.layers.N.  （如你的模型是这种结构；如果不是可注释掉）
        name = re.sub(
            r"vision_tower\.vision_model\.blocks\.(\d+)\.",
            r"vision_tower.vision_model.encoder.layers.\1.",
            name,
        )

        return name

    # === 替换原 update_weight ===
    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)."""

        # 打印一次原始键名便于排查
        if torch.distributed.get_rank() == 0:
            print(f"[vllm broadcast] update weight: {name}, dtype: {dtype}, shape: {shape}")

        # （可选）如果你暂时想跳过视觉塔，先放开这段：
        # if name.startswith("model.visual.") or name.startswith("visual.") or ".visual." in name:
        #     if torch.distributed.get_rank() == 0:
        #         print(f"[vllm broadcast] skip visual weight temporarily: {name}")
        #     return

        # 1) 做名字映射
        mapped = self._map_name_hf_to_vllm(name)

        # 2) 拉取权重（沿用你原有广播逻辑）
        if dtype != self.model_config.dtype:
            # 有些 pipeline 会出现 bf16/ fp16 不一致；与其 assert 直接卡死，不如拉取后再 cast
            # 原来是: assert dtype == self.model_config.dtype, ...
            if torch.distributed.get_rank() == 0:
                print(f"[vllm broadcast][WARN] dtype mismatch: src {dtype}, dst {self.model_config.dtype} (will cast)")
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective
            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        # 必要时做 dtype 对齐（避免后面 load_weights 因 dtype 不一致报错）
        if weight.dtype != self.model_config.dtype:
            weight = weight.to(self.model_config.dtype)

        # 3) 如果目标键不存在，给出最相近提示，然后“跳过而不是抛错”
        try:
            params_keys = [n for n, _ in self.model_runner.model.named_parameters()]
        except Exception:
            # 某些版本只能用 state_dict
            params_keys = list(self.model_runner.model.state_dict().keys())

        if mapped not in params_keys:
            # 给出 Top-3 相近键，方便你把映射表 explicit 补上
            close = get_close_matches(mapped, params_keys, n=3, cutoff=0.6)
            print(f"[vllm broadcast][WARN] key not found in vLLM: '{name}' → '{mapped}'")
            if close:
                print(f"[vllm broadcast][HINT] closest: {close}")
            # 安全跳过，避免 KeyError 直接把 job 弄挂
            del weight
            return

        # 4) 真正加载
        self.model_runner.model.load_weights(weights=[(mapped, weight)])

        del weight
        # if empty_cache:
        #     torch.cuda.empty_cache()

    # def update_weight(self, name, dtype, shape, empty_cache=False):
    #     """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
    #     if torch.distributed.get_rank() == 0:
    #         print(f"[vllm broadcast] update weight: {name}, dtype: {dtype}, shape: {shape}")

    #     assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
    #     weight = torch.empty(shape, dtype=dtype, device="cuda")
    #     if self._model_update_with_ray:
    #         import ray.util.collective as collective

    #         collective.broadcast(weight, 0, group_name=self._model_update_group)
    #     else:
    #         torch.distributed.broadcast(weight, 0, group=self._model_update_group)

    #     self.model_runner.model.load_weights(weights=[(name, weight)])

    #     del weight
    #     # TODO: should we empty cache if all weights have updated?
    #     # if empty_cache:
    #     #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
