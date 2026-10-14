"""
Lean 4 验证模块 —— 共享验证逻辑。

职责：
  1. 搜索并加载 verify_api / kimina_client 依赖
  2. 提供 verify_single() 和 verify_batch() 统一接口
  3. 暴露 HAS_VERIFIER 状态标志
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────
# 路径搜索与模块加载
# ────────────────────────────────────────────────────────────

def _append_sys_path(path_obj: Path) -> None:
    """仅在目录存在且未加入时追加到 sys.path。"""
    normalized = str(path_obj.resolve())
    if path_obj.is_dir() and normalized not in sys.path:
        sys.path.insert(0, normalized)


def _prepare_verifier_search_paths() -> None:
    """
    为 verify_api / kimina_client 准备导入路径。
    支持任意运行目录，以及脚本被挪动后的固定兜底路径。
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir.parent,
        script_dir / "scripts",
        Path("/root/pythonProject/双层链接"),
        Path("/root/pythonProject/双层连接"),
        Path("/root/pythonProject"),
    ]

    env_paths = [
        os.environ.get("VERIFY_API_DIR"),
        os.environ.get("KIMINA_CLIENT_DIR"),
        os.environ.get("VERIFY_API_PATH"),
        os.environ.get("KIMINA_CLIENT_PATH"),
    ]
    for raw in env_paths:
        if not raw:
            continue
        p = Path(raw).expanduser()
        candidates.append(p.parent if p.is_file() else p)

    for candidate in candidates:
        _append_sys_path(candidate)


def _load_verify_api_fallback():
    """当常规 import 失败时，按固定绝对路径加载 verify_api.py。"""
    candidates = [
        Path("/root/pythonProject/双层链接/verify_api.py"),
        Path("/root/pythonProject/双层连接/verify_api.py"),
        Path("/root/pythonProject/verify_api.py"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        _append_sys_path(candidate.parent)
        _append_sys_path(candidate.parent.parent)
        spec = importlib.util.spec_from_file_location("verify_api_fallback", str(candidate))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "batch_verify_lean_codes"):
            return module.batch_verify_lean_codes
    raise ImportError("无法从固定路径加载 verify_api.py")


# ────────────────────────────────────────────────────────────
# 模块级初始化：加载验证函数
# ────────────────────────────────────────────────────────────

_batch_verify_lean_codes = None
HAS_VERIFIER: bool = False

_prepare_verifier_search_paths()
try:
    from verify_api import batch_verify_lean_codes as _bvlc
    _batch_verify_lean_codes = _bvlc
    HAS_VERIFIER = True
except ImportError as e:
    try:
        _batch_verify_lean_codes = _load_verify_api_fallback()
        HAS_VERIFIER = True
        print("✅ 已通过固定绝对路径加载 Lean 4 验证模块。")
    except Exception as fallback_error:
        print(f"⚠️  警告：无法导入 Lean 4 验证模块: {e}")
        print(f"   固定路径兜底加载也失败: {fallback_error}")
        print("   已尝试脚本目录、/root/pythonProject 及环境变量路径。")
        HAS_VERIFIER = False


# ────────────────────────────────────────────────────────────
# 公共验证接口
# ────────────────────────────────────────────────────────────

def verify_batch(
    lean_codes: List[str],
    server_timeout: int = 600,
    server_url: Optional[str] = None,
) -> List[Dict]:
    """
    批量验证 Lean 4 代码。

    返回列表，每个元素为 {"pass": bool, "info": {...}}。
    若验证模块不可用，返回全部标记为失败的占位结果。
    """
    if not HAS_VERIFIER or _batch_verify_lean_codes is None:
        return [
            {
                "pass": False,
                "info": {"system_errors": "验证模块不可用", "errors": [], "warnings": [], "sorries": []},
            }
            for _ in lean_codes
        ]

    kwargs = {"server_timeout": server_timeout}
    if server_url is not None:
        kwargs["server_url"] = server_url

    results = _batch_verify_lean_codes(lean_codes, **kwargs)
    if not results or len(results) != len(lean_codes):
        raise RuntimeError("验证结果为空或数量不匹配")
    return results


def verify_single(
    lean_code: str,
    server_timeout: int = 600,
    server_url: Optional[str] = None,
) -> Tuple[bool, Optional[Dict]]:
    """
    单条验证 Lean 4 代码。

    返回 (passed, info)。验证模块不可用时返回 (True, None)。
    """
    if not HAS_VERIFIER or _batch_verify_lean_codes is None:
        return True, None

    if server_url is None:
        server_url = os.environ.get("LEAN_SERVER_URL", "http://localhost:8000")

    try:
        results = _batch_verify_lean_codes(
            [lean_code], server_timeout=server_timeout, server_url=server_url
        )
        if not results:
            return False, None
        info = results[0].get("info", {})
        passed = bool(results[0].get("pass", False))
        return passed, info
    except Exception as e:
        msg = str(e)
        if "Connection" in msg or "connect" in msg.lower():
            print(f"❌ 无法连接 Lean 验证服务器 ({server_url})")
            print("   请确保 Kimina Lean Server 已启动。")
        else:
            print(f"❌ 验证出错: {msg}")
        return False, None
