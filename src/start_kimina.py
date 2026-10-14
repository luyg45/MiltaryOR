#!/usr/bin/env python3
"""Kimina Lean Server 启动脚本 - 后台运行"""

import subprocess
import sys
import os
from pathlib import Path

# 配置
SERVER_DIR = Path.home() / "kimina-lean-server"
CLIENT_DIR = SERVER_DIR / "client"
LOG_FILE = SERVER_DIR / "server.log"


def start_server():
    # 环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CLIENT_DIR}:{env.get('PYTHONPATH', '')}"

    # nohup bash -c '...' > server.log 2>&1 &
    cmd = ["python", "-m", "server"]

    print(f"启动目录: {SERVER_DIR}")
    print(f"日志文件: {LOG_FILE}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    print("服务正在后台启动...\n")

    # 后台启动,输出重定向到日志文件
    log_f = open(LOG_FILE, "a", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(SERVER_DIR),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # 相当于 nohup,脱离父进程
    )

    print(f"已启动! PID={proc.pid}")
    print(f"查看日志: tail -f {LOG_FILE}")
    print(f"停止服务: kill {proc.pid}")


if __name__ == "__main__":
    start_server()
