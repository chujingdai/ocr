# -*- coding: utf-8 -*-
"""
OCR API 服务端
基于 PaddleOCR PP-OCRv4_mobile_rec，提供 HTTP OCR 识别接口（纯 CPU 推理）
端口: 3001
"""

import base64
import difflib
import io
import logging
import os
import time
import threading
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify, g
from werkzeug.exceptions import HTTPException

# PaddleOCR 纯 CPU 推理（PP-OCRv4_mobile_rec）

from paddleocr import PaddleOCR

app = Flask(__name__)

# ── 请求体上限（2026-10-03 新增）──────────────────────────────────────────
# 在 **Flask 读 body 之前** 就拦住超大图。
# 原链条是「读 body → base64 解码 → cv2.imdecode（★ ndarray 已分配）→ 才判
# MAX_IMAGE_PIXELS」，也就是说 4096×4096 的 50MB ndarray **已经白分配过了**；
# MAX_CONTENT_LENGTH 由 WSGI 层直接 413，比 image_pixels 校验更早、更省内存。
# 取值依据：客户端全窗图 993×433 的 PNG base64 约 0.4~1.1MB；
#   压测用的 1920×1080 纯噪声图（几乎不可压缩）base64 约 8~9MB。
#   → 20MB 有充足余量，不会误伤正常请求，又能挡住超大 body。
# 可用环境变量 OCR_MAX_BODY_MB 调整；设为 0 表示不限制。
_OCR_MAX_BODY_MB = int(os.environ.get("OCR_MAX_BODY_MB", "20"))
if _OCR_MAX_BODY_MB > 0:
    app.config["MAX_CONTENT_LENGTH"] = _OCR_MAX_BODY_MB * 1024 * 1024


@app.errorhandler(413)
def _payload_too_large(_e):
    """请求体超限：返回 JSON（与其它接口错误格式一致），便于客户端与日志识别"""
    return jsonify({
        "code": 413,
        "message": f"请求体过大（上限 {_OCR_MAX_BODY_MB}MB）",
    }), 413


# ── 并发闸门（2026-10-03 建立；OOM 事故根因修复）──────────────────────────
# 事故实证（2026-10-02 23:34，阿里云 47.121.179.65，1.6G 内存 / 2 核）：
#   客户端观测：先「连上无响应」(读超时 30s → 387s)，再「直接拒连」(10061)，
#   中断 26 分钟后才由 00:00 的宝塔每日重启计划任务拉起（服务端日志期间完全空白）。
#   内核实证：journalctl / dmesg 记录
#     'Out of memory: Killed process 2022 (python) anon-rss:1425900kB'，
#     'Free swap = 0kB' —— 即进程内存耗尽后被 OOM Killer 杀死。
#   诱因：Flask 开发服务器 threaded=True **无并发上限**，而每个请求进入 handler
#   后要先把整张图的 base64 串 + 解码缓冲拿在手里，再到 _ocr_lock 前排队 →
#   突发时排队请求无界增长 → 内存被撑爆（实测正常水位 RSS 533MB → 崩溃时 1.4G）。
# 处置：给 /ocr/* 业务接口加**并发闸门**，超限立即返回 503，不排队；
#   客户端收到非 200 会快速降级到其他服务或本地引擎，比「卡死 6 分钟」好得多。
#   /ocr/health 不受限，供守护脚本探活。
# 可用环境变量 OCR_MAX_CONCURRENT 调整（默认 3；本机 2 核建议 2~4）。
_MAX_CONCURRENT_OCR = int(os.environ.get("OCR_MAX_CONCURRENT", "3"))
_ocr_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_OCR)

# ── 只读运行状态统计（2026-10-03 新增；供 /ocr/health 上报）───────────────
# 背景：原 /ocr/health 是写死的字符串 —— 压测实证进程 RSS 1006MB、swap 换出
# 1485MB 时它依然返回 200 → watchdog 完全"探不出来"（盲区）。
# 但反过来在 health 里检查 `_ocr_lock` 又会把「正常忙碌」误判成「故障」
# （一次推理正常持锁 0.5~6.5s）→ 频繁误重启。
# ★ 约定（2026-10-03）：health 只**如实上报只读事实**，不做健康判定；
#   判定交给 watchdog，判据是「last_success_ago_s 是否持续增长」——
#   忙碌时它 ≤ 几秒，卡死时它一直涨。这是唯一能区分「卡死」与「忙碌」的指标。
_ocr_stats_lock = threading.Lock()
_ocr_active = 0                # 当前占用槽位数（只读上报）
_ocr_stat_total = 0            # 累计业务请求数
_ocr_stat_ok = 0               # 累计成功推理数
_ocr_stat_err = 0              # 累计异常数
_ocr_stat_busy = 0             # 累计"忙"拒绝数（闸门 503 + 锁超时 503）
_ocr_last_success_ts = 0.0     # 最近一次**成功**推理的 time.monotonic()；0 = 启动后从未成功
_ocr_last_success_wall = ""    # 同上，墙上时间字符串（便于人读）

# 推理锁等待上限（秒）。取 30 与客户端 `OCR_API_TIMEOUT = 30` 对齐：
# 超过客户端自身超时的等待已无意义（客户端早已降级到别的服务/本地引擎），
# 继续占着槽位只会连累其它设备。可用环境变量 OCR_LOCK_TIMEOUT 调整。
_OCR_LOCK_TIMEOUT = float(os.environ.get("OCR_LOCK_TIMEOUT", "30"))


class OcrBusy(RuntimeError):
    """推理锁等待超时（上游持锁线程可能已卡死）

    由各路由转成 503 返回，客户端会快速降级。
    ⚠️ **绝不据此重建引擎** —— 详见 `_infer_with_retry` 的说明。
    """


@app.before_request
def _acquire_ocr_slot():
    """业务接口取槽位；取不到直接 503（排队 = 内存无界增长）"""
    global _ocr_active, _ocr_stat_total, _ocr_stat_busy
    p = request.path or ""
    if p.startswith("/ocr/") and p != "/ocr/health":
        with _ocr_stats_lock:
            _ocr_stat_total += 1
        if not _ocr_slots.acquire(blocking=False):
            with _ocr_stats_lock:
                _ocr_stat_busy += 1
            return jsonify({
                "code": 503,
                "message": "服务繁忙（并发上限 %d），请稍后重试" % _MAX_CONCURRENT_OCR,
            }), 503
        g._ocr_slot_held = True
        with _ocr_stats_lock:
            _ocr_active += 1


@app.after_request
def _release_ocr_slot(resp):
    global _ocr_active
    if getattr(g, "_ocr_slot_held", False):
        g._ocr_slot_held = False
        _ocr_slots.release()
        with _ocr_stats_lock:
            _ocr_active -= 1
    return resp

# ── 业务日志（2026-09-08 按设备分文件夹；2026-09-14 改为按天滚动 ocr_api_YYYYMMDD.log）──
_log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(_log_base, exist_ok=True)
_log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
_device_loggers = {}
_device_loggers_lock = threading.Lock()
_OCR_LOG_RETENTION_DAYS = 0   # 2026-09-14 用户要求：取消 7 天自动删除，日志手动清理；0=永久保留

def _cleanup_old_ocr_logs():
    """启动时清理超过保留天数的 ocr_api_*.log（按文件名中的 8 位日期判断）

    2026-09-14：日志已移入 logs/<device>/debug/，同时兼容旧路径 logs/<device>/。
    2026-09-14 用户要求：保留天数设为 0 = 永久保留，不再自动删除（手动清理）。
    """
    if _OCR_LOG_RETENTION_DAYS <= 0:
        return
    from datetime import datetime as _dt, timedelta as _td
    import re as _re
    try:
        cutoff = _dt.now().date() - _td(days=_OCR_LOG_RETENTION_DAYS)
        removed = 0
        scan_dirs = []
        for _dev in os.listdir(_log_base):
            _dev_dir = os.path.join(_log_base, _dev)
            if not os.path.isdir(_dev_dir):
                continue
            scan_dirs.append(_dev_dir)                      # 兼容旧位置
            scan_dirs.append(os.path.join(_dev_dir, 'debug'))  # 新位置
        for _dir in scan_dirs:
            if not os.path.isdir(_dir):
                continue
            for _fn in os.listdir(_dir):
                if not (_fn.startswith('ocr_api') and _fn.endswith('.log')):
                    continue
                _m = _re.search(r'(\d{8})', _fn)
                if not _m:
                    os.remove(os.path.join(_dir, _fn))   # 旧版无日期 ocr_api.log，直接删
                    removed += 1
                    continue
                if _dt.strptime(_m.group(1), '%Y%m%d').date() < cutoff:
                    os.remove(os.path.join(_dir, _fn))
                    removed += 1
        if removed:
            print(f"[OCR日志] 已清理 {removed} 个超过 {_OCR_LOG_RETENTION_DAYS} 天的旧日志")
    except Exception:
        pass

_cleanup_old_ocr_logs()


def _get_device_logger(device):
    """按设备+日期返回独立 logger（logs/<device>/debug/ocr_api_YYYYMMDD.log，一天一个文件）

    2026-09-14 用户要求：
      · 日志一天一个：原固定文件名 ocr_api.log 无限追加且不分天 → 按天命名；
      · 目录分层：OCR 日志属调试产物 → 移入 logs/<device>/debug/ 子目录。
    缓存 key 含日期，跨天自动创建新 handler 写入新文件。
    """
    device = str(device).strip() if device else "unknown"
    today = time.strftime('%Y%m%d')
    key = (device, today)
    with _device_loggers_lock:
        if key not in _device_loggers:
            # 2026-10-03：淘汰**同一设备**历史日期的 logger。
            # 原实现只增不减，key=(device, 日期) → 长跑时每天每设备多一个
            # logger + 一个常开的 FileHandler → fd 与内存缓慢泄漏。
            # 新的一天首次写日志时顺手关掉该设备更早的 handler（释放 fd）。
            # 注意：只在锁内操作，且只动"同设备、非今天"的项，不影响其它设备/今天的写入。
            for _stale_key in [k for k in _device_loggers if k[0] == device and k != key]:
                _stale = _device_loggers.pop(_stale_key)
                for _h in list(_stale.handlers):
                    try:
                        _h.close()
                    except Exception:
                        pass
                    _stale.removeHandler(_h)
            log_dir = os.path.join(_log_base, device, 'debug')
            os.makedirs(log_dir, exist_ok=True)
            logger = logging.getLogger(f'ocr_api_{device}_{today}')
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = logging.FileHandler(os.path.join(log_dir, f'ocr_api_{today}.log'), encoding='utf-8')
            handler.setFormatter(_log_formatter)
            logger.addHandler(handler)
            _device_loggers[key] = logger
        return _device_loggers[key]

# 推理串行锁（防止多请求并发导致 CPU 争抢超时）
# 2026-10-03：不再直接用 `with _ocr_lock:`，统一走 `_infer_with_retry()`
#   —— 那里改用 acquire(timeout=_OCR_LOCK_TIMEOUT)，超时抛 OcrBusy 转 503。
#   背景：普通 Lock 无超时，一旦某次推理卡死，锁被永久占用 → 3 个槽位会被
#   3 个"在锁上干等"的请求逐个占满 → 之后所有请求全部 503，且旧版 /ocr/health
#   仍返回 200 → watchdog 看不见 → 永久假死，只能人工重启。
_ocr_lock = threading.Lock()
# 2026-09-11 实测调优（monitor/test/bench_ocr_threads.py，960x433 真实截图 36 框，每档 2 次）：
#   intra=2 → 1465 / 1104 ms
#   intra=4 → 1081 /  846 ms
#   intra=6 →  663 /  626 ms   ★ 最优
#   intra=8 →  860 /  773 ms（超订反而变慢）
# OCR 耗时 ≈ 识别文本框数 × 每框 rec 时间（服务端日志实测 3 框=511ms、84 框=6482ms），
# 故该参数对「长安/傲来摆摊广告密集」场景收益最直接，84 框预计从 6482ms 降到 ~3.6s。
# 若出现 PaddleOCR "CUDA/CPU allocation error" 或与其他 5 个监控进程抢 CPU，可用环境变量调小：
#   set OCR_INTRA_THREADS=2 && python monitor/api-new/server.py
_OCR_INTRA_THREADS = int(os.environ.get("OCR_INTRA_THREADS", "6"))
_ocr_engine = None

def _get_ocr_engine():
    """获取/初始化 PaddleOCR PP-OCRv4 引擎"""
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR-v4] 正在加载 PaddleOCR PP-OCRv4 引擎（首次冷加载约 15~60s，请等待）...")
        _t0 = time.time()
        _ocr_engine = PaddleOCR(
            lang='ch',
            ocr_version='PP-OCRv4',
            use_angle_cls=False,
            use_gpu=False,
            show_log=False,
            cpu_threads=_OCR_INTRA_THREADS,
            )
        print("[OCR-v4] PaddleOCR PP-OCRv4 引擎已加载（耗时 %.1fs）" % (time.time() - _t0))
    return _ocr_engine
def _reinit_ocr_engine():
    """重新初始化 OCR 引擎（内存分配失败后调用）"""
    global _ocr_engine
    try:
        del _ocr_engine
    except Exception:
        pass
    _ocr_engine = None
    import gc
    gc.collect()
    print("[OCR-v4] 引擎已释放，等待下次请求重新初始化")


def _paddle_result_to_common(paddle_result):
    if not paddle_result or not paddle_result[0]:
        return []
    formatted = []
    for line in paddle_result[0]:
        box = line[0]
        text, score = line[1]
        formatted.append({
            "box": [[float(p[0]), float(p[1])] for p in box],
            "text": text,
            "score": float(score),
        })
    return formatted

# 最大图像尺寸限制（防止超大图片导致 OOM）
MAX_IMAGE_PIXELS = 4096 * 4096


def decode_image(image_b64):
    """将 Base64 编码字符串解码为 BGR 图像数组"""
    if not image_b64:
        return None, "缺少 image 参数"
    try:
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None, "无法解码图像，请确保为有效的 PNG/JPEG Base64 编码"
        h, w = img.shape[:2]
        if w * h > MAX_IMAGE_PIXELS:
            return None, f"图像过大 ({w}x{h})，超过最大限制"
        return img, None
    except Exception as e:
        return None, f"Base64 解码失败: {str(e)}"


def crop_region(img, region):
    """裁剪图像区域 [x, y, w, h]"""
    if region is None:
        return img
    try:
        x, y, w, h = int(region[0]), int(region[1]), int(region[2]), int(region[3])
        fh, fw = img.shape[:2]
        x = max(0, min(x, fw))
        y = max(0, min(y, fh))
        w = max(0, min(w, fw - x))
        h = max(0, min(h, fh - y))
        if w > 0 and h > 0:
            return img[y:y + h, x:x + w]
        return img
    except (TypeError, ValueError):
        return img


def preprocess_for_ocr(img):
    """图像预处理：灰度化 + 直方图均衡化"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _infer_with_retry(img):
    """在 `_ocr_lock` 保护下执行一次推理 —— **全部业务路径的唯一入口**

    行为（与改造前一致，只多了「超时脱身」）：
      · 锁等待超过 `_OCR_LOCK_TIMEOUT` → 抛 `OcrBusy`（路由转 503）
      · `CUDA/CPU allocation error` / `PaddleOCRError` → 重建引擎后重试一次
      · 其它异常原样抛出（路由转 500）

    2026-10-03 改造说明
    -------------------
    ① **原来只有 `ocr_image` 带 bad-allocation 重试，`ocr_image_scaled`
       （`/ocr/fast`、`/ocr/keywords/fast` 两条路径）没有** → 那两条路径遇到内存
       分配失败无法自愈。现在统一走本函数。
    ② **锁等待超时只抛 `OcrBusy`，绝不重建引擎**：
       超时意味着持锁线程**仍在用旧引擎推理**（它就是卡住了才会超时）。此时若执行
       `del _ocr_engine; _ocr_engine = None`，旧推理 Session 会被 GC，而卡住的
       线程仍在它的 C++ 原生内存上跑 → **use-after-free 段错误**，比「卡住」更糟
       （卡住至少进程还活着、watchdog 能探到；段错误会连带所有在途请求一起死）。
       因此重建**只**发生在 `CUDA/CPU allocation error` 路径：那里引擎确实已异常，且由
       **持有锁的线程自己**调用，无竞态 —— 这是既有正确行为，不要改。
    ③ 每次成功推理都更新 `_ocr_last_success_ts` —— 这是 `/ocr/health` 用来区分
       「忙碌」与「卡死」的唯一指标（忙碌时 ≤ 几秒，卡死时持续增长）。
    """
    global _ocr_stat_ok, _ocr_stat_err, _ocr_stat_busy
    global _ocr_last_success_ts, _ocr_last_success_wall

    if not _ocr_lock.acquire(timeout=_OCR_LOCK_TIMEOUT):
        with _ocr_stats_lock:
            _ocr_stat_busy += 1
        print(f"[OCR-v4] 推理锁等待超过 {_OCR_LOCK_TIMEOUT:.0f}s（上游线程可能卡死）"
              f" → 返回 503，不重建引擎")
        raise OcrBusy("等待识别引擎超过 %.0fs" % _OCR_LOCK_TIMEOUT)
    try:
        try:
            result = _get_ocr_engine().ocr(img, cls=False)
        except Exception as _e:
            _err_msg = str(_e)
            if "CUDA/CPU allocation error" in _err_msg or "PaddleOCRError" in type(_e).__name__:
                print(f"[OCR-v4] 内存分配失败 → 重建引擎重试: {_err_msg[:120]}")
                _reinit_ocr_engine()
                try:
                    result = _get_ocr_engine().ocr(img, cls=False)
                except Exception as _e2:
                    print(f"[OCR-v4] 重建后仍失败: {str(_e2)[:120]}")
                    with _ocr_stats_lock:
                        _ocr_stat_err += 1
                    raise
            else:
                with _ocr_stats_lock:
                    _ocr_stat_err += 1
                raise
        with _ocr_stats_lock:
            _ocr_stat_ok += 1
            _ocr_last_success_ts = time.monotonic()
            _ocr_last_success_wall = time.strftime("%Y-%m-%d %H:%M:%S")
        return _paddle_result_to_common(result)
    finally:
        _ocr_lock.release()


def ocr_image(img):
    """执行 OCR 识别，返回格式化结果列表"""
    result = _infer_with_retry(img)
    formatted = []
    if result:
        for line in result:
            box = line[0]
            text = line[1]
            score = line[2]
            formatted.append({
                "box": [[float(p[0]), float(p[1])] for p in box],
                "text": text,
                "score": float(score),
            })
    return formatted


def ocr_image_scaled(img, scale):
    """降采样 OCR，坐标映射回原图"""
    fh, fw = img.shape[:2]
    if 0 < scale < 1.0:
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        if nw < fw or nh < fh:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        nw, nh = fw, fh
    scale_x = fw / nw
    scale_y = fh / nh

    # 2026-10-03：改为与 ocr_image 共用 _infer_with_retry（原实现裸调引擎，
    # CUDA/CPU allocation error 时无法自愈，与 ocr_image 行为不一致）
    result = _infer_with_retry(img)
    formatted = []
    if result:
        for line in result:
            box = line[0]
            text = line[1]
            score = line[2]
            box_scaled = [[float(p[0] * scale_x), float(p[1] * scale_y)] for p in box]
            formatted.append({
                "box": box_scaled,
                "text": text,
                "score": float(score),
            })
    return formatted


# ═══════════════════════════════════════════════════════════════
#  POST /ocr/full
# ═══════════════════════════════════════════════════════════════
@app.route("/ocr/full", methods=["POST"])
def ocr_full():
    t0 = time.time()
    try:
        body = request.get_json(silent=True)
        device = (body or {}).get("device", "")
        logger = _get_device_logger(device)
        if not body:
            logger.warning("/ocr/full 请求体为空")
            return jsonify({"code": 400, "message": "请求体为空或非 JSON 格式"}), 400

        img, err = decode_image(body.get("image", ""))
        if err:
            logger.warning("/ocr/full 图像解码失败: %s", err)
            return jsonify({"code": 400, "message": err}), 400

        region = body.get("region")
        img = crop_region(img, region)
        # 2026-09-18 问题211：preprocess 改为调用方可选，默认 True（既有调用方零影响）。
        # 灰度化+直方图均衡化在**暗场彩色画面**（战斗场景，平均亮度≈24）下会把
        # 彩色小字的对比度抹平、同时放大背景噪声 → 实测单位名识别数 4 → 0。
        # 战斗首轮全窗 OCR 传 preprocess=False（保持原图彩色）后恢复正常。
        if body.get("preprocess", True):
            img = preprocess_for_ocr(img)
        results = ocr_image(img)

        elapsed_ms = int((time.time() - t0) * 1000)
        texts = [r["text"] for r in results]
        logger.info("/ocr/full region=%s elapsed=%dms count=%d texts=%s",
                         region, elapsed_ms, len(results), texts)
        return jsonify({
            "code": 0,
            "message": "ok",
            "data": {
                "results": results,
                "elapsed_ms": elapsed_ms,
            }
        })
    except OcrBusy as _busy:
        # 2026-10-03：推理锁等待超时 → 503（客户端会快速降级到别的服务/本地引擎）。
        # 只打 stdout，不写设备日志 —— 高并发时避免刷屏。
        print(f"  [OCR-v4] /ocr/full 返回 503：{_busy}")
        return jsonify({"code": 503,
                        "message": "识别引擎繁忙（等待超时 %ds），请稍后重试"
                                   % int(_OCR_LOCK_TIMEOUT)}), 503
    except HTTPException:
        # 2026-10-03 实测发现：MAX_CONTENT_LENGTH 触发的 `RequestEntityTooLarge`(413)
        # 是在 `request.get_json()` 读流时抛出的 → 它是 HTTPException 子类 →
        # 若直接落到下面的 `except Exception`，会被吞成 **500**，`@app.errorhandler(413)`
        # 永远不会生效（实测确实返回了 500）。这里必须原样上抛交给 errorhandler。
        raise
    except Exception:
        traceback.print_exc()
        _get_device_logger("unknown").error("/ocr/full 异常", exc_info=True)
        return jsonify({"code": 500, "message": "服务端内部错误"}), 500


# ═══════════════════════════════════════════════════════════════
#  POST /ocr/fast
# ═══════════════════════════════════════════════════════════════
@app.route("/ocr/fast", methods=["POST"])
def ocr_fast():
    t0 = time.time()
    try:
        body = request.get_json(silent=True)
        device = (body or {}).get("device", "")
        logger = _get_device_logger(device)
        if not body:
            logger.warning("/ocr/fast 请求体为空")
            return jsonify({"code": 400, "message": "请求体为空或非 JSON 格式"}), 400

        img, err = decode_image(body.get("image", ""))
        if err:
            logger.warning("/ocr/fast 图像解码失败: %s", err)
            return jsonify({"code": 400, "message": err}), 400

        region = body.get("region")
        scale = body.get("scale", 0.5)

        img = crop_region(img, region)
        img = preprocess_for_ocr(img)
        results = ocr_image_scaled(img, scale)

        elapsed_ms = int((time.time() - t0) * 1000)
        texts = [r["text"] for r in results]
        logger.info("/ocr/fast region=%s scale=%.2f elapsed=%dms count=%d texts=%s",
                         region, scale, elapsed_ms, len(results), texts)
        return jsonify({
            "code": 0,
            "message": "ok",
            "data": {
                "results": results,
                "elapsed_ms": elapsed_ms,
            }
        })
    except OcrBusy as _busy:
        print(f"  [OCR-v4] /ocr/fast 返回 503：{_busy}")
        return jsonify({"code": 503,
                        "message": "识别引擎繁忙（等待超时 %ds），请稍后重试"
                                   % int(_OCR_LOCK_TIMEOUT)}), 503
    except HTTPException:
        # 2026-10-03 实测发现：MAX_CONTENT_LENGTH 触发的 `RequestEntityTooLarge`(413)
        # 是在 `request.get_json()` 读流时抛出的 → 它是 HTTPException 子类 →
        # 若直接落到下面的 `except Exception`，会被吞成 **500**，`@app.errorhandler(413)`
        # 永远不会生效（实测确实返回了 500）。这里必须原样上抛交给 errorhandler。
        raise
    except Exception:
        traceback.print_exc()
        _get_device_logger("unknown").error("/ocr/fast 异常", exc_info=True)
        return jsonify({"code": 500, "message": "服务端内部错误"}), 500


# ═══════════════════════════════════════════════════════════════
#  POST /ocr/keywords
# ═══════════════════════════════════════════════════════════════
@app.route("/ocr/keywords", methods=["POST"])
def ocr_keywords():
    t0 = time.time()
    try:
        body = request.get_json(silent=True)
        device = (body or {}).get("device", "")
        logger = _get_device_logger(device)
        if not body:
            logger.warning("/ocr/keywords 请求体为空")
            return jsonify({"code": 400, "message": "请求体为空或非 JSON 格式"}), 400

        img, err = decode_image(body.get("image", ""))
        if err:
            logger.warning("/ocr/keywords 图像解码失败: %s", err)
            return jsonify({"code": 400, "message": err}), 400

        keywords = body.get("keywords", [])
        if not keywords:
            logger.warning("/ocr/keywords 缺少 keywords 参数")
            return jsonify({"code": 400, "message": "缺少 keywords 参数"}), 400

        confidence = body.get("confidence", 0.6)
        region = body.get("region")

        img = crop_region(img, region)
        img = preprocess_for_ocr(img)
        results = ocr_image(img)

        matches = []
        for item in results:
            if item["score"] < confidence:
                continue
            text = item["text"]
            box = item["box"]
            cx = sum(p[0] for p in box) / len(box)
            cy = sum(p[1] for p in box) / len(box)
            for kw in keywords:
                if kw in text:
                    matches.append({
                        "text": text,
                        "keyword": kw,
                        "center": [float(cx), float(cy)],
                        "box": box,
                        "score": item["score"],
                    })

        elapsed_ms = int((time.time() - t0) * 1000)
        match_texts = [f"{m['keyword']}→{m['text']}" for m in matches]

        # ══ 2026-09-24 用户要求：匹配失败时输出「条件下相似度」诊断 ══
        # 原日志只记 ocr_count/match_count，失败时**看不到 OCR 到底读到了什么**
        # → 无法区分两种失败：① 文字被遮挡/太小根本没识别出来（ocr_count 骤降）；
        # ② 识别出来了但读成错字（如「已拥有」→「已拥身」）。
        # 实测（设备1 20260924 凌波城家具交付）：失败帧 count=4
        #   texts=['任务','师门任务','买到白鹤展翅帘','的事：']   ← 「请选择要做的事」
        # 选择框遮挡任务面板，「已拥有」整行没进 OCR 结果；
        # 而框消失后同一调用 count=7 且含「已拥有）送给师傅」→ 匹配成功。
        # → 失败时打印**全部 OCR 文本** + 与各关键词的最高相似度 Top5（滑动窗口
        #   字符级相似度，能区分「已拥有」/「已拥身」这类同长错字），
        #   并把 near_misses 一并返回给客户端，让 monitor 日志也能看到。
        near_misses = []
        _ocr_texts = []
        if not matches and results:
            def _best_sim(kw, text):
                """关键词与文本的最佳窗口相似度（滑动窗口逐字对比）"""
                n = len(kw)
                if not n or not text:
                    return 0.0
                if n >= len(text):
                    return difflib.SequenceMatcher(None, kw, text).ratio()
                _best = 0.0
                for _i in range(len(text) - n + 1):
                    _r = difflib.SequenceMatcher(None, kw, text[_i:_i + n]).ratio()
                    if _r > _best:
                        _best = _r
                        if _best >= 0.999:
                            break
                return _best

            # ★ 2026-10-03 性能优化（**等价变换，不改变任何输出**）：
            #   两个字符串**没有任何共同字符**时 `SequenceMatcher.ratio()` 必然为 0，
            #   而下方只保留 `similarity > 0` 的候选 → 可用 set 交集做廉价前置过滤，
            #   直接跳过最贵的「滑动窗口逐字对比」。代价 O(len) 换来省掉 O(len²) 次
            #   SequenceMatcher 构造。
            #   为什么必须廉价化：失败诊断（`not matches and results`）正是**最常见**
            #   的路径（问题311 就是），84 个文本框 × 10 个关键词 ≈ 几千次纯 Python
            #   相似度计算。这段在 `_ocr_lock` 之外（✅ 不占锁），但会长时间持有 GIL
            #   → 在 2 核机器上会拖慢并发推理。
            _kw_charsets = [(kw, set(kw)) for kw in keywords]

            _cands = []
            for item in results:
                _t = (item.get("text") or "").strip()
                if not _t:
                    continue
                _t_set = set(_t)
                _bk, _bs = "", 0.0
                for kw, _kw_chars in _kw_charsets:
                    if not (_kw_chars & _t_set):
                        continue        # 无共同字符 → 相似度必为 0（等价短路）
                    _r = _best_sim(kw, _t)
                    if _r > _bs:
                        _bs, _bk = _r, kw
                _cands.append((_bs, _bk, _t, item.get("score", 0.0)))
            _cands.sort(key=lambda c: c[0], reverse=True)
            # 只保留相似度 > 0 的候选（sim=0.00 表示该文本与所有关键词毫无共同
            # 字符，属纯噪声，不列入）——这样「候选为空」= 画面完全没有相关文字
            # （遮挡/字号过小），与「有候选但读成错字」可一眼区分。
            near_misses = [
                {"similarity": round(s, 3), "keyword": k, "text": t,
                 "ocr_score": round(float(sc), 3)}
                for s, k, t, sc in _cands[:5] if s > 0
            ]
            # 2026-09-24 追加：把 OCR 实读文本回传客户端——客户端原来只能猜
            # （near_misses 为空时打印「OCR 未读到任何文本」），实际可能读到了
            # 别的字（ocr_count>0）→ 终端信息与事实不符（用户实测反馈）。
            _ocr_texts = [(r.get("text") or "") for r in results][:40]
            logger.info(
                "/ocr/keywords ✗ 未匹配 → ocr_texts(%d)=%s | 相似度Top5=%s",
                len(results), [r.get("text", "") for r in results],
                [f"{n['similarity']:.2f}「{n['text']}」(kw={n['keyword']},"
                 f"ocr={n['ocr_score']:.2f})" for n in near_misses])

        logger.info("/ocr/keywords region=%s keywords=%s conf=%.2f elapsed=%dms ocr_count=%d match_count=%d matches=%s",
                         region, keywords, confidence, elapsed_ms, len(results), len(matches), match_texts)
        return jsonify({
            "code": 0,
            "message": "ok",
            "data": {
                "matches": matches,
                "elapsed_ms": elapsed_ms,
                "near_misses": near_misses,
                "ocr_count": len(results),
                "ocr_texts": _ocr_texts,
            }
        })
    except OcrBusy as _busy:
        print(f"  [OCR-v4] /ocr/keywords 返回 503：{_busy}")
        return jsonify({"code": 503,
                        "message": "识别引擎繁忙（等待超时 %ds），请稍后重试"
                                   % int(_OCR_LOCK_TIMEOUT)}), 503
    except HTTPException:
        # 2026-10-03 实测发现：MAX_CONTENT_LENGTH 触发的 `RequestEntityTooLarge`(413)
        # 是在 `request.get_json()` 读流时抛出的 → 它是 HTTPException 子类 →
        # 若直接落到下面的 `except Exception`，会被吞成 **500**，`@app.errorhandler(413)`
        # 永远不会生效（实测确实返回了 500）。这里必须原样上抛交给 errorhandler。
        raise
    except Exception:
        traceback.print_exc()
        _get_device_logger("unknown").error("/ocr/keywords 异常", exc_info=True)
        return jsonify({"code": 500, "message": "服务端内部错误"}), 500


# ═══════════════════════════════════════════════════════════════
#  POST /ocr/keywords/fast
# ═══════════════════════════════════════════════════════════════
@app.route("/ocr/keywords/fast", methods=["POST"])
def ocr_keywords_fast():
    t0 = time.time()
    try:
        body = request.get_json(silent=True)
        device = (body or {}).get("device", "")
        logger = _get_device_logger(device)
        if not body:
            logger.warning("/ocr/keywords/fast 请求体为空")
            return jsonify({"code": 400, "message": "请求体为空或非 JSON 格式"}), 400

        img, err = decode_image(body.get("image", ""))
        if err:
            logger.warning("/ocr/keywords/fast 图像解码失败: %s", err)
            return jsonify({"code": 400, "message": err}), 400

        keywords = body.get("keywords", [])
        if not keywords:
            logger.warning("/ocr/keywords/fast 缺少 keywords 参数")
            return jsonify({"code": 400, "message": "缺少 keywords 参数"}), 400

        confidence = body.get("confidence", 0.6)
        region = body.get("region")
        scale = body.get("scale", 0.5)

        img = crop_region(img, region)
        img = preprocess_for_ocr(img)
        results = ocr_image_scaled(img, scale)

        matches = []
        for item in results:
            if item["score"] < confidence:
                continue
            text = item["text"]
            box = item["box"]
            cx = sum(p[0] for p in box) / len(box)
            cy = sum(p[1] for p in box) / len(box)
            for kw in keywords:
                if kw in text:
                    matches.append({
                        "text": text,
                        "keyword": kw,
                        "center": [float(cx), float(cy)],
                        "box": box,
                        "score": item["score"],
                    })

        elapsed_ms = int((time.time() - t0) * 1000)
        match_texts = [f"{m['keyword']}→{m['text']}" for m in matches]
        logger.info("/ocr/keywords/fast region=%s keywords=%s conf=%.2f scale=%.2f elapsed=%dms ocr_count=%d match_count=%d matches=%s",
                         region, keywords, confidence, scale, elapsed_ms, len(results), len(matches), match_texts)
        return jsonify({
            "code": 0,
            "message": "ok",
            "data": {
                "matches": matches,
                "elapsed_ms": elapsed_ms,
            }
        })
    except OcrBusy as _busy:
        print(f"  [OCR-v4] /ocr/keywords/fast 返回 503：{_busy}")
        return jsonify({"code": 503,
                        "message": "识别引擎繁忙（等待超时 %ds），请稍后重试"
                                   % int(_OCR_LOCK_TIMEOUT)}), 503
    except HTTPException:
        # 2026-10-03 实测发现：MAX_CONTENT_LENGTH 触发的 `RequestEntityTooLarge`(413)
        # 是在 `request.get_json()` 读流时抛出的 → 它是 HTTPException 子类 →
        # 若直接落到下面的 `except Exception`，会被吞成 **500**，`@app.errorhandler(413)`
        # 永远不会生效（实测确实返回了 500）。这里必须原样上抛交给 errorhandler。
        raise
    except Exception:
        traceback.print_exc()
        _get_device_logger("unknown").error("/ocr/keywords/fast 异常", exc_info=True)
        return jsonify({"code": 500, "message": "服务端内部错误"}), 500


# ═══════════════════════════════════════════════════════════════
#  GET/POST /ocr/health  (健康检查)
# ═══════════════════════════════════════════════════════════════
@app.route("/ocr/health", methods=["GET", "POST"])
def health():
    """健康检查 —— 2026-10-03 改为**上报只读真实状态**（原来只返回写死的字符串）

    ★ 为什么原来那个是坏的（已实证）
      压测最凶的一刻：进程 RSS 1006MB、VmSwap 1485MB、系统 swap 用到 2197MB，
      原 `health()` 依然返回 `{"code":0,"status":"running"}` → **watchdog 完全探不出来**。
      于是下面这些情况服务实际已废，但每分钟探活的 watchdog 永远不会重启它：
        · PaddleOCR 引擎初始化失败（业务请求全 500）→ health 仍 200
        · 推理卡死（请求全挂到超时）          → health 仍 200
        · `_ocr_lock` 被卡死线程永久持有       → 之后全 503，health 仍 200

    ★ 为什么这里**刻意不做健康判定**
      在 health 里检查 `_ocr_lock` 是否可用，会把**正常忙碌**误判成**故障**
      （一次推理正常持锁 0.5~6.5s）→ watchdog 频繁误重启。
      「卡死」与「忙碌」本质是**时间维度**的判断，应该由 watchdog 累计比较。

    ★ 正确的判据（交给 watchdog）
      看 `last_success_ago_s` **是否持续增长**：
        · 忙碌/正常 → 该值始终 ≤ 几秒（刚成功过）
        · 卡死      → 该值只增不减（例如 > 180s 无成功推理 → 判定故障 → 重启）
      这比「连续 N 次 503」更直接：卡死时可能连 503 都返不出来（请求一直挂着）。
    """
    now = time.monotonic()
    with _ocr_stats_lock:
        _active = _ocr_active
        _tot, _ok, _err, _busy = (_ocr_stat_total, _ocr_stat_ok,
                                  _ocr_stat_err, _ocr_stat_busy)
        _last_ts, _last_wall = _ocr_last_success_ts, _ocr_last_success_wall
    return jsonify({
        "code": 0,
        "message": "ok",
        "data": {
            "status": "running",
            "pid": os.getpid(),
            # 引擎是否已实例化（懒加载：进程刚启动时 False 属正常）
            "engine": "PP-OCRv4_mobile_rec (PaddleOCR)",
            "engine_ready": _ocr_engine is not None,
            "active_slots": _active,
            "max_concurrent": _MAX_CONCURRENT_OCR,
            # 只读事实：True 既可能是「正在推理」也可能是「卡死」，
            # 与 last_success_ago_s 一起看即可区分（不要单独拿它判定健康）
            "infer_lock_held": _ocr_lock.locked(),
            # ★ 关键指标：距最近一次**成功**推理过了多久（None = 进程启动后从未成功）
            "last_success_ago_s": (round(now - _last_ts, 1) if _last_ts else None),
            "last_success_at": _last_wall or None,
            "stat": {"total": _tot, "ok": _ok, "err": _err, "busy": _busy},
        }
    })


if __name__ == "__main__":
    # 2026-09-11：端口改为可配置，便于不打断线上服务时另起实例做线程数对比实测
    _PORT = int(os.environ.get("OCR_API_PORT", "3001"))
    print("OCR API 服务启动中...")
    print(f"端口: {_PORT}   引擎: PP-OCRv4_mobile_rec   CPU线程数={_OCR_INTRA_THREADS}")
    print(f"并发闸门: {_MAX_CONCURRENT_OCR}   推理锁等待上限: {_OCR_LOCK_TIMEOUT:.0f}s"
          f"   请求体上限: {(_OCR_MAX_BODY_MB or '不限')}MB")
    print("接口: /ocr/full, /ocr/fast, /ocr/keywords, /ocr/keywords/fast")
    # 2026-10-03：显式声明 threaded=True。
    # 实测当前 Flask 3.0 默认即为并发（8 并发 → 3×200 + 5×503），但**依赖框架
    # 默认值有隐患**（换版本/换部署方式可能退回串行）→ 显式写死，行为可预期。
    # 注意：并发数由 `_ocr_slots` 闸门（默认 3）控制，threaded 只是"允许并发"。
    app.run(host="0.0.0.0", port=_PORT, debug=False, threaded=True)