"""
全局配置文件 —— 所有可调参数集中管理。

使用方式：
  from config import CFG
  print(CFG.MODEL_PATH)
"""


class _Config:
    """项目全局配置，修改此处即可影响所有模块。"""

    # ═══════════════════════════════════════════════════════════
    # 模型路径（做对照实验改这里即可）
    # ───────────────────────────────────────────────────────────
    # 切换基座模型示例：
    #   求解层 GGUF：SOLVER_MODEL_PATH = "/root/prover-v2/Goedel-Prover-V2-32B.gguf"
    #   求解层 HF ：SOLVER_MODEL_PATH = "/root/.cache/modelscope/hub/models/Goedel-LM/Goedel-Prover-V2-8B"
    #   建模层 HF ：MODELING_MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/main"
    # ═══════════════════════════════════════════════════════════

    # ── 求解层模型 ──
    # 支持两种格式：① GGUF 单文件（量化、用 llama_cpp）  ② HuggingFace safetensors 目录（用 transformers）
    SOLVER_MODEL_PATH = "/root/.cache/modelscope/hub/models/Goedel-LM/Goedel-Prover-V2-8B/"
    # 后端选择：
    #   "auto"         → 路径以 .gguf 结尾走 llama_cpp，否则走 transformers（推荐）
    #   "llama_cpp"    → 强制 GGUF（须配合 .gguf 路径）
    #   "transformers" → 强制 HF safetensors（须配合 HF 目录路径）
    SOLVER_BACKEND = "auto"

    # ── 建模层模型（始终是 HuggingFace safetensors 目录）──
    MODELING_MODEL_DIR = "/root/pythonProject/huggingface_cache/models--Buchilaguo--ATF-8B/snapshots/f2c0e32190550e9ce947a25374536e188fb829f8"

    # ═══════════════════════════════════════════════════════════
    # 求解层参数
    # ═══════════════════════════════════════════════════════════
    SOLVER_MAX_RETRIES = 7          # 求解最大重试轮次
    SOLVER_MAX_TOKENS = 4096        # 单次生成最大 token 数
    SOLVER_TEMPERATURE = 0.1        # 初始采样温度
    SOLVER_TEMP_INCREMENT = 0.15      # 每轮温度递增步长
    SOLVER_TEMP_CAP = 0.7             # 温度上限
    SOLVER_REPEAT_PENALTY = 1.2     # 重复惩罚（>1 惩罚重复 token，防止退化循环）
    SOLVER_N_CTX = 40960            # 模型上下文窗口（= 训练上限 n_ctx_train）
    SOLVER_N_GPU_LAYERS = -1        # GPU 层数（-1 = 全部）
    SOLVER_GPU_IDS = [3, 4]         # 求解层专用 GPU 列表（与建模层完全不重叠）
    SOLVER_MAX_MEM_PER_GPU = 8      # 求解层每卡最大占用 GiB（8B fp16≈16GB 分2卡，每卡~8G权重，剩余给attention）
    SOLVER_MAX_SEQ_LEN = 4096       # 求解层输入最大 token 数（超出截断，防 SDPA OOM）
    SOLVER_SAMPLING_BUDGET = 4      # 采样预算：每道题独立生成多少个候选证明（Pass@k 中的 k）

    # ═══════════════════════════════════════════════════════════
    # 建模层参数
    # ═══════════════════════════════════════════════════════════
    MODELING_MAX_RETRIES = 3        # 建模验证最大重试轮次
    MODELING_MAX_NEW_TOKENS = 2048  # 建模生成最大 token
    MODELING_TEMPERATURE = 0.6      # 建模采样温度
    MODELING_BATCH_SIZE = 2         # 批量大小（A30 14.56GiB 建议 ≤2，防 SDPA OOM）
    MODELING_GPU_IDS = [0, 1, 2]    # 建模层专用 GPU 列表（T4 单卡 15GiB，8B模型需 2~3 卡）
    MODELING_MAX_MEM_PER_GPU = 11   # 建模层每卡最大占用 GiB（T4 15GiB，剩余留给 attention）
    MODELING_MAX_SEQ_LEN = 4096     # 建模层输入最大 token 数（超出截断，防 attention OOM）

    # ═══════════════════════════════════════════════════════════
    # Lean 验证参数
    # ═══════════════════════════════════════════════════════════
    LEAN_SERVER_TIMEOUT = 600       # Lean 验证服务器超时（秒）
    LEAN_MAX_HEARTBEATS = 200000    # Lean 验证 maxHeartbeats（默认 200000）

    # ═══════════════════════════════════════════════════════════
    # 文件路径
    # ═══════════════════════════════════════════════════════════
    DEFAULT_DATASET = "/root/pythonProject/连贯作业1/ADFP_100.jsonl"
    DEFAULT_SUMMARY_FILE = "/root/pythonProject/连贯作业1/batch_summary.json"
    DEFAULT_INPUT_FILE = "/root/pythonProject/双层链接/artifacts/input.lean"


# 单例导出
CFG = _Config()
