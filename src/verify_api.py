import os
import json
from typing import Dict, List, Optional
from kimina_client import KiminaClient, Snippet


def format_results(results):
    """
    将 CheckResponse.results (List[ReplResponse]) 转换为易读的字典列表。

    处理三种情况：
    1. ReplResponse.error 不为空 → 服务器级错误（连接超时、REPL 异常等）
    2. ReplResponse.response 为 Error 类型 → 含 'message' 键的 REPL 级错误
    3. ReplResponse.response 为 CommandResponse → 正常的 Lean 编译结果
    """
    returns = []

    for repl_result in results:
        format_res = {
            'info': {
                'system_errors': None,
                'sorries': [],
                'errors': [],
                'warnings': []
            }
        }

        # 情况 1：服务器级错误（如连接超时、REPL 异常）
        if repl_result.error:
            format_res['info']['system_errors'] = repl_result.error
            format_res['pass'] = False
            returns.append(format_res)
            continue

        response = repl_result.response

        # 情况 2：响应为空（理论不应发生，但做防御性处理）
        if response is None:
            format_res['info']['system_errors'] = "Empty response from Lean server"
            format_res['pass'] = False
            returns.append(format_res)
            continue

        # 情况 3：Error 类型（仅有 'message' 键，无 'messages' 列表）
        if 'message' in response:
            format_res['info']['system_errors'] = response.get('message', str(response))
            format_res['pass'] = False
            returns.append(format_res)
            continue

        # 情况 4：正常的 CommandResponse（有 'messages' 和可选 'sorries'）
        sorries = response.get('sorries')
        format_res['info']['sorries'] = list(sorries) if sorries else []

        messages = response.get('messages')
        if messages:
            for item in messages:
                severity = item.get('severity', '')
                if severity == 'warning':
                    format_res['info']['warnings'].append(item)
                elif severity == 'error':
                    format_res['info']['errors'].append(item)

        format_res['pass'] = len(format_res['info']['errors']) == 0
        returns.append(format_res)

    return returns


def batch_verify_lean_codes(
    lean_contents: List[str],
    concurrency: int = 5,
    server_timeout: int = 600,
    server_url: Optional[str] = None
) -> List[Dict]:
    """
    批量验证 Lean 4 代码。

    参数:
        lean_contents: Lean 4 代码片段列表
        concurrency: 并发批次数
        server_timeout: 每个请求的超时时间（秒）
        server_url: Lean 服务器地址，默认从 LEAN_SERVER_URL 环境变量读取，
                    若未设置则回退到 http://localhost:8000
    """
    if server_url is None:
        server_url = os.environ.get("LEAN_SERVER_URL", "http://localhost:8000")

    client = KiminaClient(api_url=server_url, http_timeout=server_timeout)

    snips = [Snippet(id=f"{idx}", code=str(proof)) for idx, proof in enumerate(lean_contents)]

    response = client.check(snips=snips, max_workers=concurrency)

    # 按 id 排序以保证结果顺序与输入一致
    sorted_results = sorted(response.results, key=lambda r: int(r.id))
    results = format_results(sorted_results)

    return results
