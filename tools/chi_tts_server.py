"""带鉴权与限流的小叽 TTS HTTP 服务 (包装上游 api_v2, 不修改其代码).

在 api_v2 全部端点前加一层校验:
  - API Key: 环境变量 CHI_TTS_API_KEY (必填, 未设置则拒绝启动).
    客户端通过 `Authorization: Bearer <key>` 或 `?api_key=<key>` 提供.
  - 限流: /tts 每 IP 每分钟最多 CHI_TTS_RATE_LIMIT 次 (默认 30, 设 0 关闭).

参数与 api_v2.py 完全一致:
  python tools/chi_tts_server.py -c GPT_SoVITS/configs/tts_infer_chi.yaml -a 0.0.0.0 -p 9880

注意: 明文 HTTP 下 API Key 会被中间人嗅探, 仅适合内网/低风险公网场景;
面向公众分发应用时, 建议由后端服务代为调用, 不要把唯一密钥嵌进客户端.
"""

import os
import sys
import time
from collections import defaultdict, deque

API_KEY = os.environ.get("CHI_TTS_API_KEY", "")
if not API_KEY:
    sys.exit("[错误] 未设置 CHI_TTS_API_KEY 环境变量, 拒绝以无鉴权方式启动")

RATE_LIMIT = int(os.environ.get("CHI_TTS_RATE_LIMIT", "30"))

import uvicorn  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import api_v2  # noqa: E402  (模块级完成参数解析与模型加载)

APP = api_v2.APP

_hits: dict[str, deque] = defaultdict(deque)


def _extract_key(request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("api_key", "")


@APP.middleware("http")
async def auth_and_rate_limit(request, call_next):
    if _extract_key(request) != API_KEY:
        return JSONResponse(status_code=401, content={"message": "invalid or missing api key"})

    if RATE_LIMIT > 0 and request.url.path == "/tts":
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse(status_code=429, content={"message": "rate limit exceeded"})
        q.append(now)

    return await call_next(request)


if __name__ == "__main__":
    host = api_v2.args.bind_addr
    if host == "None":  # 与上游一致: -a None 监听双栈
        host = None
    uvicorn.run(app=APP, host=host, port=api_v2.args.port, workers=1)
