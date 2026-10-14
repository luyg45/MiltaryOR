"""
求解层 GPU 子进程 Worker —— 在独立 GPU 上运行 llama_cpp 推理。

由 solver.py 中的 _SolverLLMProxy 自动启动，通过 stdin/stdout JSON 管道通信。
本脚本不应手动运行。

协议：
  请求 (stdin, 每行一个 JSON):
    {"cmd": "generate", "messages": [...], "max_tokens": 4096, "temperature": 0.1, "repeat_penalty": 1.2}
    {"cmd": "reset"}
    {"cmd": "shutdown"}

  响应 (stdout, 每行一个 JSON):
    {"status": "ok", "result": {...}}
    {"status": "error", "error": "..."}
"""

import sys
import os
import json

# ── 关键：在任何 CUDA 相关库导入之前设置 GPU 隔离 ──
# GPU ID 由父进程通过环境变量 SOLVER_WORKER_GPU_ID 传入
_gpu_id = os.environ.get("SOLVER_WORKER_GPU_ID", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id

# Windows 终端 UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _send_response(obj: dict):
    """向父进程发送 JSON 响应（单行，flush）。"""
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    from config import CFG

    # 延迟导入 llama_cpp（此时 CUDA_VISIBLE_DEVICES 已生效）
    from llama_cpp import Llama

    model_path = CFG.SOLVER_MODEL_PATH
    sys.stderr.write(f"[solver_worker] GPU={_gpu_id} 加载模型: {model_path}\n")
    sys.stderr.flush()

    llm = Llama(
        model_path=model_path,
        n_ctx=CFG.SOLVER_N_CTX,
        n_gpu_layers=CFG.SOLVER_N_GPU_LAYERS,
        verbose=False,
    )
    sys.stderr.write("[solver_worker] 模型加载完成，等待请求...\n")
    sys.stderr.flush()

    # 通知父进程：就绪
    _send_response({"status": "ready"})

    # ── 主循环：逐行读取请求 ──
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _send_response({"status": "error", "error": f"JSON 解析失败: {e}"})
            continue

        cmd = req.get("cmd", "")

        if cmd == "shutdown":
            _send_response({"status": "ok", "result": "bye"})
            break

        elif cmd == "reset":
            llm.reset()
            _send_response({"status": "ok", "result": None})

        elif cmd == "generate":
            try:
                result = llm.create_chat_completion(
                    messages=req["messages"],
                    max_tokens=req.get("max_tokens", 4096),
                    temperature=req.get("temperature", 0.1),
                    repeat_penalty=req.get("repeat_penalty", 1.0),
                )
                _send_response({"status": "ok", "result": result})
            except Exception as e:
                _send_response({"status": "error", "error": f"{type(e).__name__}: {e}"})

        else:
            _send_response({"status": "error", "error": f"未知命令: {cmd}"})

    sys.stderr.write("[solver_worker] 已退出\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
