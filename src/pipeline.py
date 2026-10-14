"""
Pipeline 主入口 —— 双层连接系统（建模 → 验证 → 求解）。

用法：
    python pipeline.py [--input-file ...] [--batch-size 4] ...
"""

from modeling import parse_args, run_batch_pipeline


def main() -> int:
    args = parse_args()
    run_batch_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
