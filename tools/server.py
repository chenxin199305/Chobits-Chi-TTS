"""带鉴权与限流的小叽 TTS HTTP 服务 (包装上游 api_v2, 不修改其代码).

在 api_v2 全部端点前加一层校验:
  - API Key: 环境变量 CHII_TTS_API_KEY (必填, 未设置则拒绝启动).
    客户端通过 `Authorization: Bearer <key>` 或 `?api_key=<key>` 提供.
  - 限流: /tts 与 /v1/audio/speech 每 IP 每分钟最多 CHII_TTS_RATE_LIMIT 次 (默认 60, 设 0 关闭).
  - TLS: 同时设置 CHII_TTS_SSL_CERTFILE / CHII_TTS_SSL_KEYFILE 时以 HTTPS 启动.

环境变量已从 CHI_TTS_* 改名为 CHII_TTS_* (角色官方罗马字 Chii);
读取时优先新名, 读不到回退旧名 CHI_TTS_*, 老部署无需改动即可继续运行.

另提供 OpenAI TTS 兼容垫片 (客户端 baseUrl 填 http(s)://<IP>:9880/v1):
  - GET  /v1/models        → 固定返回 chii-tts
  - POST /v1/audio/speech  → OpenAI TTS 协议: {"model", "input", "voice", "response_format"?, "speed"?}
    voice 映射服务端参考音频 (当前仅 "chii"; 旧名 "chi"/"chi-default" 作为兼容别名归一到 "chii");
    response_format 支持 wav/aac/opus
    (默认 wav; mp3/flac/pcm 暂不支持, 返回 400); speed (0.25~4.0) 映射 speed_factor;
    text_lang 由服务端钉死 (auto, 不支持则回退 ja), 其余采样参数取 api_v2 默认值.
    wav 为流式输出 (streaming_mode=2, 边合成边推流); aac/opus 为一次性返回.

参数与 api_v2.py 完全一致:
  python tools/server.py -c GPT_SoVITS/configs/tts_infer_chii.yaml -a 0.0.0.0 -p 9880

注意: 未启用 TLS 时 API Key 明文传输, 仅适合内网/低风险公网场景;
面向公众分发应用时, 建议由后端服务代为调用, 不要把唯一密钥嵌进客户端.
"""

import json
import os
import sys
import time
from collections import defaultdict, deque


def _getenv(suffix: str, default: str = "") -> str:
    """读取 CHII_TTS_<suffix>, 未设置时回退旧名 CHI_TTS_<suffix> (老部署兼容)."""
    return os.environ.get(f"CHII_TTS_{suffix}") or os.environ.get(f"CHI_TTS_{suffix}", default)


API_KEY = _getenv("API_KEY")
if not API_KEY:
    sys.exit("[错误] 未设置 CHII_TTS_API_KEY 环境变量 (旧名 CHI_TTS_API_KEY 也可), 拒绝以无鉴权方式启动")

RATE_LIMIT = int(_getenv("RATE_LIMIT", "60"))

# TLS: 两个变量都设置时以 HTTPS 启动 (自签名证书见 README「部署为 HTTP 服务」)
SSL_CERTFILE = _getenv("SSL_CERTFILE")
SSL_KEYFILE = _getenv("SSL_KEYFILE")
if bool(SSL_CERTFILE) != bool(SSL_KEYFILE):
    sys.exit("[错误] CHII_TTS_SSL_CERTFILE 与 CHII_TTS_SSL_KEYFILE 必须同时设置")

import uvicorn  # noqa: E402
from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import api_v2  # noqa: E402  (模块级完成参数解析与模型加载)

APP = api_v2.APP

_hits: dict[str, deque] = defaultdict(deque)

# --- OpenAI TTS 兼容垫片 -------------------------------------------------
# voice → 服务端参考音频映射 (ref_audio_path 是服务器本地路径, 客户端不可见);
# 可用 CHII_TTS_REF_AUDIO / CHII_TTS_REF_TEXT_FILE 覆盖默认参考音频与参考文本 (旧名 CHI_TTS_* 回退兼容)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ref_text_file = _getenv("REF_TEXT_FILE", os.path.join(REPO_ROOT, "models", "ref_text.txt"))
try:
    with open(_ref_text_file, encoding="utf-8") as _f:
        _ref_text = _f.read().strip()
except OSError:
    _ref_text = ""

VOICES = {
    "chii": {
        "ref_audio_path": _getenv("REF_AUDIO", os.path.join(REPO_ROOT, "models", "ref_audio.wav")),
        "prompt_text": _ref_text,
        "prompt_lang": "ja",
    },
}
# 旧客户端兼容别名: 在 voice 校验前归一到新音色名
VOICE_ALIASES = {"chi": "chii", "chi-default": "chii"}
TTS_MODEL = "chii-tts"
# OpenAI response_format → api_v2 media_type (ogg 即 opus 的 ogg 封装);
# mp3/flac/pcm 暂不支持
FORMAT_MAP = {"wav": "wav", "aac": "aac", "opus": "ogg"}
# OpenAI TTS 协议不传语言: 钉死 auto (不支持 auto 的旧版本回退 ja)
TEXT_LANG = "auto" if "auto" in api_v2.tts_config.languages else "ja"


def _openai_error(code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={
        "error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}})


@APP.get("/v1/models")
async def openai_models():
    return {"object": "list", "data": [
        {"id": TTS_MODEL, "object": "model", "created": 0, "owned_by": "chobits-chii"}]}


@APP.post("/v1/audio/speech")
async def openai_audio_speech(request: Request):
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return _openai_error(400, "请求体不是合法 JSON")
    text = str(body.get("input") or "").strip()
    if not text:
        return _openai_error(400, "input 不能为空")
    voice = str(body.get("voice") or "chii")
    voice = VOICE_ALIASES.get(voice, voice)
    if voice not in VOICES:
        return _openai_error(400, f"voice: {voice} 不存在, 可用: {sorted(VOICES)}")
    fmt = str(body.get("response_format") or "wav").lower()
    if fmt not in FORMAT_MAP:
        return _openai_error(400, f"response_format: {fmt} 暂不支持, 可用: {sorted(FORMAT_MAP)}")
    speed = body.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or not 0.25 <= float(speed) <= 4.0:
        return _openai_error(400, "speed 须在 0.25~4.0 之间")

    v = VOICES[voice]
    req = api_v2.TTS_Request().dict()  # 其余参数取 api_v2 默认值
    req.update({
        "text": text,
        "text_lang": TEXT_LANG,
        "ref_audio_path": v["ref_audio_path"],
        "prompt_text": v["prompt_text"],
        "prompt_lang": v["prompt_lang"],
        "media_type": FORMAT_MAP[fmt],
        "speed_factor": float(speed),
        # wav 走流式 (边合成边推流: 首块 WAV 头 + 后续 raw PCM, 首字延迟低);
        # aac/opus 逐块编码会拼出损坏帧, 保持非流式一次性返回
        "streaming_mode": 2 if FORMAT_MAP[fmt] == "wav" else False,
    })
    return await api_v2.tts_handle(req)

# ------------------------------------------------------------------------


def _extract_key(request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("api_key", "")


@APP.middleware("http")
async def auth_and_rate_limit(request, call_next):
    if _extract_key(request) != API_KEY:
        return JSONResponse(status_code=401, content={"message": "invalid or missing api key"})

    if RATE_LIMIT > 0 and request.url.path in ("/tts", "/v1/audio/speech"):
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
    uvicorn.run(
        app=APP,
        host=host,
        port=api_v2.args.port,
        workers=1,
        ssl_certfile=SSL_CERTFILE or None,
        ssl_keyfile=SSL_KEYFILE or None,
    )
