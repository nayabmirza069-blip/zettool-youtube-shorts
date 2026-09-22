#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZetTool YouTube Shorts Agent  (v2)
End-to-end pipeline for one daily 9:16 Short:

  Gemini script with a strong hook (finance/calculator/productivity niche)
  -> Pexels stock background video behind big text cards (fallback: gradient cards)
  -> Edge TTS voiceover + gentle zoom + crossfade transitions
  -> optional YouTube Data API upload

Nothing is uploaded unless youtube_config.json has upload.enabled=true AND a
valid token exists (run  python youtube_agent.py --auth  once).

Run:
  python youtube_agent.py             generate + upload (if enabled/token)
  python youtube_agent.py --no-upload generate the Short only
  python youtube_agent.py --auth      one-time YouTube OAuth login
  python youtube_agent.py --report    show status + recent videos
  python youtube_agent.py --force     ignore the daily limit (testing)
"""

import asyncio
import datetime as dt
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import edge_tts
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_CONFIG = os.path.join(BASE_DIR, "config.json")
AGENT_CONFIG = os.path.join(BASE_DIR, "youtube_config.json")
LOG_PATH = os.path.join(BASE_DIR, "youtube_agent.log")
STATE_PATH = os.path.join(BASE_DIR, "_youtube_state.json")
BUILD_DIR = os.path.join(BASE_DIR, "shorts_work")
OUT_DIR = os.path.join(BASE_DIR, "shorts")
TOKEN_PATH = os.path.join(BASE_DIR, "youtube_token.json")
SECRET_PATH = os.path.join(BASE_DIR, "client_secret.json")

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# ZetTool brand palette (matches style.css)
GRAD_TOP = (15, 23, 42)
GRAD_BOTTOM = (30, 27, 75)
ACCENT = (99, 102, 241)
TEXT_WHITE = (248, 250, 252)
TEXT_MUTED = (148, 163, 184)
TEXT_SOFT = (165, 180, 252)
PILL_BG = (49, 46, 129)
END_TOP = (79, 70, 229)
END_BOTTOM = (236, 72, 153)

FONT_DIR = "C:/Windows/Fonts"
LINUX_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def log(msg):
    line = "[{}] {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            log("ERROR: missing file: " + path)
            sys.exit(1)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state():
    if not os.path.exists(STATE_PATH):
        return {"used_topics": {}, "daily_counts": {}, "videos": [], "last_run": ""}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(st):
    st["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
    st["videos"] = st.get("videos", [])[-30:]
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Gemini (same pattern as the other agents: retries + saturation short-circuit)
GEMINI_SATURATED = {"flag": False}


def gemini_json(gcfg, prompt, retries=2):
    if GEMINI_SATURATED["flag"]:
        return None
    api_key = gcfg.get("gemini_api_key")
    model = gcfg.get("gemini_model", "gemini-3.6-flash")
    if not api_key:
        return None
    url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent".format(model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 3000, "responseMimeType": "application/json"},
    }
    for i in range(retries):
        try:
            r = requests.post(
                url + "?key=" + api_key,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if r.status_code == 200:
                data = r.json()
                txt = data["candidates"][0]["content"]["parts"][0]["text"]
                return parse_json_loose(txt)
            log("Gemini HTTP {}: {}".format(r.status_code, r.text[:160]))
            if r.status_code in (429, 500, 503):
                GEMINI_SATURATED["flag"] = True
                break
        except Exception as e:
            log("Gemini attempt {} error: {}".format(i + 1, e))
            GEMINI_SATURATED["flag"] = True
            break
        time.sleep(4 + i * 4)
    return None


def parse_json_loose(txt):
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Script generation
SYSTEM_SEED_TOPICS = [
    "how to read your salary breakup CTC gross in hand",
    "how to calculate EMI before you take a home loan",
]

SCRIPT_PROMPT = """You write scripts for a faceless viral-style YouTube Short about personal finance, calculators and simple online tools.
Audience: everyday Indian people who want the exact number, fast. Tone: confident, conversational, human.
Topic: {topic}

Allowed tool pages (pick the ONE that fits best):
{tool_urls}

Return ONLY valid JSON, no markdown fences:
{{"title": "<SEO title: main keyword naturally included, under 60 chars, click-worthy and honest>", "segments": [{{"text": "<one short spoken line, 6-13 words>", "visual": "<2-4 word Pexels video search keyword, e.g. 'money counting hands', 'office papers', 'calculator closeup'>"}}], "cta": "<short spoken call-to-action, max 2 sentences, mentions {site_name}>", "description": "<2-3 sentence useful description; first sentence includes the main keyword and mentions India when relevant; later includes the exact tool_url>", "tags": ["<SEO tags: 2 specific topic keywords + 2 broad, lower-case alphanumeric>"], "tool_url": "<one exact URL from the allowed list>"}}

HARD RULES:
- SEGMENTS[0] IS THE HOOK: one short impactful line (6-10 words) with a surprising number, a contrast everyone believes, or an emotional claim that stops the scroll. Examples: 'The middle class pays the MOST tax. Heres why.' / '90% of people read this salary line wrong.' / 'Your bank wants you to miss this 5-second check.'
- Educational and honest. No get-rich-quick, no guaranteed returns, no paid/grey-hat talk.
- 4-6 segments; total spoken words across segments 55-85.
- Every line short and natural to speak out loud. Numbers punchy.
- The cta sounds helpful, not spammy, and must invite the free tool.
- description contains tool_url exactly as it appears in the allowed list.
- tags lower-case alphanumeric only, 4-6 of them, no spaces."""


def pick_topic(acfg, state, today):
    pool = acfg.get("seed_topics") or SYSTEM_SEED_TOPICS
    if os.environ.get("YT_DETERMINISTIC_TOPIC") == "1":
        idx = (dt.date.today() - dt.date(2026, 1, 1)).days % len(pool)
        return pool[idx]
    used = state.get("used_topics", {})
    fresh = [t for t in pool if t not in used]
    if not fresh:
        old = dt.date.today() - dt.timedelta(days=30)
        fresh = [t for t in pool if used.get(t, "") < old.isoformat()]
        if not fresh:
            fresh = pool
    return random.choice(fresh)


TAG_MAP = {
    "salary-calculator.html": ["salary", "ctc", "inhand"],
    "loan-calculator.html": ["loan", "emi", "homefinance"],
    "gst-calculator.html": ["gst", "tax", "india"],
    "income-tax-calculator.html": ["incometax", "taxsaving", "india"],
    "gpa-calculator.html": ["gpa", "college", "education"],
    "cgpa-to-percentage.html": ["cgpa", "percentage", "college"],
    "percentage-calculator.html": ["percentage", "math", "fastmath"],
    "fuel-cost-calculator.html": ["fuel", "roadtrip", "travel"],
    "sleep-calculator.html": ["sleep", "health", "routine"],
    "mileage-calculator.html": ["mileage", "fuel", "car"],
    "electricity-bill-calculator.html": ["electricity", "saving", "bill"],
    "fd-calculator.html": ["fd", "deposit", "savings"],
    "date-calculator.html": ["date", "days", "calculator"],
    "age-calculator.html": ["age", "birthday", "fun"],
    "bmi-calculator.html": ["bmi", "health", "weight"],
    "tip-calculator.html": ["tip", "bills", "travel"],
    "word-to-pdf.html": ["pdftools", "converter", "productivity"],
    "merge-pdf.html": ["pdf", "pdfmerge", "productivity"],
    "split-pdf.html": ["pdf", "pdfsplit", "productivity"],
    "compress-pdf.html": ["pdf", "compress", "email"],
    "protect-pdf.html": ["pdf", "privacy", "security"],
    "rotate-pdf.html": ["pdf", "rotate", "fixpdf"],
    "image-to-text.html": ["ocr", "scanner", "text"],
    "word-counter.html": ["wordcount", "writing", "productivity"],
    "case-converter.html": ["caseconverter", "writing", "typing"],
    "number-to-words.html": ["numbers", "cheque", "indian"],
    "text-to-speech.html": ["texttospeech", "audio", "learning"],
    "online-notepad.html": ["notepad", "notes", "privacy"],
    "jpg-to-pdf.html": ["jpg", "pdf", "converter"],
    "pdf-to-jpg.html": ["pdftojpg", "pdf", "images"],
    "couple-compatibility-quiz.html": ["love", "couple", "funquiz"],
}


def tags_for_tool(tool_url, fallback):
    for slug, tags in TAG_MAP.items():
        if slug in tool_url:
            return tags[:4]
    return fallback


def parse_segments(data, acfg):
    segs = []
    for s in data["segments"]:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text", "")).strip()
        if not (0 < len(text.split()) <= 18):
            continue
        segs.append({
            "text": text,
            "visual": str(s.get("visual", "")).strip() or "money finance",
        })
        if len(segs) >= acfg.get("max_segments", 6):
            break
    return segs


def build_script(scfg, acfg, topic):
    tool_urls = acfg.get("tool_urls") or [acfg.get("featured_tool_url", "https://www.zettool.com/")]
    prompt = SCRIPT_PROMPT.format(
        topic=topic,
        tool_urls="\n".join("- " + u for u in tool_urls),
        site_name=scfg.get("site_name", "ZetTool"),
    )
    data = gemini_json(scfg, prompt, retries=2)
    if data and isinstance(data, dict) and data.get("segments"):
        segs = parse_segments(data, acfg)
        if len(segs) >= acfg.get("min_segments", 4):
            tool_url = str(data.get("tool_url", "")).strip()
            if tool_url not in tool_urls:
                tool_url = tool_urls[0]
            return {
                "title": str(data.get("title", "")).strip()[:70] or topic.title(),
                "segments": segs,
                "cta": str(data.get("cta", "")).strip(),
                "description": str(data.get("description", "")).strip(),
                "tags": [re.sub(r"[^a-z0-9]", "", str(t).lower()) for t in (data.get("tags") or []) if re.sub(r"[^a-z0-9]", "", str(t).lower())][:12],
                "tool_url": tool_url,
                "source": "gemini",
            }
    return fallback_script(scfg, acfg, topic)


def pick_tool_for_topic(topic, tool_urls):
    keys = [
        ("salary", "salary-calculator.html"),
        ("emi", "loan-calculator.html"),
        ("loan", "loan-calculator.html"),
        ("gst", "gst-calculator.html"),
        ("tax", "income-tax-calculator.html"),
        ("gpa", "gpa-calculator.html"),
        ("cgpa", "cgpa-to-percentage.html"),
        ("percent", "percentage-calculator.html"),
        ("fuel", "fuel-cost-calculator.html"),
        ("sleep", "sleep-calculator.html"),
        ("mileage", "mileage-calculator.html"),
        ("electricity", "electricity-bill-calculator.html"),
        ("deposit", "fd-calculator.html"),
        ("offer letter", "income-tax-calculator.html"),
        ("date", "date-calculator.html"),
        ("age", "age-calculator.html"),
        ("bmi", "bmi-calculator.html"),
        ("tip", "tip-calculator.html"),
        ("word to pdf", "word-to-pdf.html"),
        ("words to pdf", "word-to-pdf.html"),
        ("ocr", "image-to-text.html"),
        ("scan", "image-to-text.html"),
        ("compress", "compress-pdf.html"),
        ("password", "protect-pdf.html"),
        ("protect", "protect-pdf.html"),
        ("split", "split-pdf.html"),
        ("merge", "merge-pdf.html"),
        ("word count", "word-counter.html"),
        ("count word", "word-counter.html"),
        ("case", "case-converter.html"),
        ("number to words", "number-to-words.html"),
        ("text to speech", "text-to-speech.html"),
        ("notepad", "online-notepad.html"),
        ("pdf", "merge-pdf.html"),
    ]
    tl = topic.lower()
    for kw, slug in keys:
        if kw in tl:
            url = next((u for u in tool_urls if slug in u), None)
            if url:
                return url
    return tool_urls[0]


def fallback_script(scfg, acfg, topic):
    core = topic.strip().strip(".")
    seq = core if core.lower().startswith("how ") else "How to " + core
    title = (seq[:1].upper() + seq[1:] + " | 30-Sec Guide")[:58]
    tool_urls = acfg.get("tool_urls") or [acfg.get("featured_tool_url", "https://www.zettool.com/")]
    tool_url = pick_tool_for_topic(core, tool_urls)
    short = core if len(core) <= 42 else (core[:39].rsplit(" ", 1)[0] + "...")
    hook = ("Most people guess '{}' - and get it wrong.".format(short)) if len(core) <= 42 else "Most people get this wrong. Here is the 30-second fix."
    segments = [
        {"text": hook, "visual": "money finance counting"},
        {"text": "You do not need apps. You do not need an account.", "visual": "calculator closeup"},
        {"text": "Enter the details. Get the exact answer in seconds.", "visual": "coins savings jar"},
        {"text": "Do it once, and you will never do it by hand again.", "visual": "papers documents desk"},
        {"text": "The tool is free - link is in the description.", "visual": "smartphone notes"},
    ]
    site_name = scfg.get("site_name", "ZetTool")
    tags = tags_for_tool(tool_url, ["calculator", "shorts"])
    for extra in ("shorts", "free"):
        if extra not in tags and len(tags) < 6:
            tags.append(extra)
    return {
        "title": title,
        "segments": segments,
        "cta": "Try it free on {} - the exact tool page is in the description. No sign-up needed. Thanks for watching!".format(site_name),
        "description": "A quick 30-second explainer of {}. Free online tool: {}. No sign-up, works on your phone and laptop.".format(core, tool_url),
        "tags": tags,
        "tool_url": tool_url,
        "source": "template",
    }


# ---------------------------------------------------------------------------
# Card / overlay rendering
def font(name, size):
    cands = [os.path.join(FONT_DIR, name)]
    if not os.path.exists(cands[0]):
        cands += LINUX_FONTS
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def gradient_base(w, h, top, bottom):
    base = Image.new("RGB", (1, h))
    px = base.load()
    for y in range(h):
        t = y / max(1, h - 1)
        px[0, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return base.resize((w, h))


def draw_text_center(draw, cx, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)
    return y + fnt.size


def draw_pill(draw, cx, cy, text, fnt, fill, bg, pad_h=22, radius=42):
    w = draw.textlength(text, font=fnt)
    h = fnt.size + 20
    x0 = cx - (w + pad_h * 2) / 2
    x1 = cx + (w + pad_h * 2) / 2
    y0 = cy - h / 2
    y1 = cy + h / 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg)
    draw.text(((x0 + x1) / 2 - w / 2, y0 + 10), text, font=fnt, fill=fill)


def wrap_text(draw, text, fnt, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=fnt) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_big_font(d, text, size, max_w, max_h):
    fnt = font("segoeuib.ttf", size)
    lines = wrap_text(d, text, fnt, max_w)
    while sum(fnt.size + 16 for _ in lines) > max_h and size > 52:
        size -= 6
        fnt = font("segoeuib.ttf", size)
        lines = wrap_text(d, text, fnt, max_w)
    return fnt, lines


def draw_content(d, text, idx, total, w, h, is_hook, alpha):
    f_small_b = font("segoeuib.ttf", 34)
    f_small = font("segoeui.ttf", 30)
    f_cap = font("segoeuib.ttf", 54)

    if is_hook:
        draw_pill(d, w / 2, 130, "THE 30-SECOND FIX", f_small_b, (255, 255, 255), (244, 63, 94))
    else:
        draw_pill(d, w / 2, 130, "ZETTOOL.COM", f_small_b, TEXT_SOFT, PILL_BG)

    max_w = int(w * 0.80)
    lines = wrap_text(d, text, f_cap, max_w)
    if len(lines) > 2:
        f_cap = font("segoeuib.ttf", 46)
        lines = wrap_text(d, text, f_cap, max_w)
    lh = f_cap.size + 18
    max_line_w = 0
    for ln in lines:
        lw = d.textlength(ln, font=f_cap)
        if lw > max_line_w:
            max_line_w = lw
    pad = 36
    band_w = int(max_line_w) + pad * 2
    band_h = lh * len(lines) + 40
    by = h - 340
    if by + band_h > h - 190:
        by = h - 190 - band_h
    bx = (w - band_w) / 2
    if alpha:
        d.rounded_rectangle([bx, by, bx + band_w, by + band_h], radius=30, fill=(7, 10, 25, 170))
    else:
        d.rounded_rectangle([bx, by, bx + band_w, by + band_h], radius=30, fill=(7, 10, 25))
    ty = by + 22
    for ln in lines:
        d.text((bx + pad, ty), ln, font=f_cap, fill=TEXT_WHITE)
        ty += lh

    bar_w = 540
    bar_y = h - 140
    bx0 = (w - bar_w) / 2
    d.rounded_rectangle([bx0, bar_y, bx0 + bar_w, bar_y + 12], radius=6, fill=(45, 55, 85, 180) if alpha else (45, 55, 85))
    fill_w = int(bar_w * (idx + 1) / max(total, 1))
    d.rounded_rectangle([bx0, bar_y, bx0 + fill_w, bar_y + 12], radius=6, fill=ACCENT)
    draw_text_center(d, w / 2, h - 92, "zettool.com", f_small_b, TEXT_MUTED)


def render_segment_overlay(text, idx, total, w, h, is_hook=False):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_content(ImageDraw.Draw(img), text, idx, total, w, h, is_hook, alpha=True)
    return img


def render_segment_card(text, idx, total, w, h, is_hook=False):
    img = gradient_base(w, h, GRAD_TOP, GRAD_BOTTOM)
    draw_content(ImageDraw.Draw(img), text, idx, total, w, h, is_hook, alpha=False)
    return img


def render_end_card(w, h):
    img = gradient_base(w, h, END_TOP, END_BOTTOM)
    d = ImageDraw.Draw(img)
    f_mid_b = font("segoeuib.ttf", 64)
    f_big = font("segoeuib.ttf", 118)
    f_small = font("segoeui.ttf", 44)

    draw_pill(d, w / 2, int(h * 0.30), "MADE WITH ZETTOOL", f_mid_b, (255, 255, 255), (30, 27, 75))
    y = draw_text_center(d, w / 2, int(h * 0.46), "Free tools.", f_big, (255, 255, 255))
    y = draw_text_center(d, w / 2, y, "No sign-up.", f_big, (255, 255, 255))
    draw_text_center(d, w / 2, int(h * 0.72), "zettool.com", font("segoeuib.ttf", 92), (255, 255, 255))
    draw_text_center(d, w / 2, int(h * 0.72) + 120, "Calculators  \u2022  PDF tools  \u2022  Text tools", f_small, (237, 233, 254))
    draw_text_center(d, w / 2, h - 200, "Works on phone & laptop", font("segoeui.ttf", 36), (233, 213, 255))
    return img


# ---------------------------------------------------------------------------
# Pexels stock footage
def pick_pexels_file(video):
    files = [
        f for f in video.get("video_files", [])
        if f.get("height") and f["height"] >= f.get("width", 0)
        and f.get("file_type") == "video/mp4"
        and f.get("width", 0) >= 480
    ]
    files.sort(key=lambda f: (-{"hd": 2, "sd": 1}.get(f.get("quality"), 0), -f["width"]))
    return files[0] if files else None


def pexels_clip(api_key, query, run_dir, idx, site_name):
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "orientation": "portrait", "per_page": 3},
            headers={"Authorization": api_key},
            timeout=25,
        )
        if r.status_code != 200:
            log("Pexels HTTP {} for '{}'".format(r.status_code, query))
            return None
        for v in r.json().get("videos", []):
            f = pick_pexels_file(v)
            if not f:
                continue
            try:
                dl = requests.get(f["link"], timeout=60)
                if dl.status_code != 200 or len(dl.content) < 200000:
                    continue
                p = os.path.join(run_dir, "bg_{}.mp4".format(idx))
                with open(p, "wb") as fh:
                    fh.write(dl.content)
                log("Footage: '{}' -> {}x{} {} ({}s)".format(query, f["width"], f["height"], f.get("quality", "?"), v.get("duration", "?")))
                return p
            except Exception as e:
                log("Footage download error: {}".format(e))
                continue
    except Exception as e:
        log("Pexels error: {}".format(e))
    return None


# ---------------------------------------------------------------------------
# Video assembly (footage or cards, ken burns zoom, crossfades)
def audio_duration(path):
    try:
        with AudioFileClip(path) as a:
            return float(a.duration)
    except Exception:
        return 0.0


def cover_crop(clip, w, h):
    cw, ch = clip.w, clip.h
    s = max(float(w) / cw, float(h) / ch)
    nw, nh = int(cw * s), int(ch * s)
    clip = clip.resized((nw, nh))
    x = max(0, int((nw - w) / 2))
    y = max(0, int((nh - h) / 2))
    return clip.with_effects([vfx.Crop(x1=x, y1=y, width=w, height=h)])


def loop_to_clip(bg_path, dur):
    a = VideoFileClip(bg_path)
    if a.duration is None or a.duration <= 0:
        return a
    if a.duration >= dur:
        return a
    n = int(math.ceil(dur / a.duration))
    repeats = [VideoFileClip(bg_path) for _ in range(n)]
    return concatenate_videoclips(repeats, method="compose").with_duration(dur)


def kenburns(clip):
    d = clip.duration or 3.0
    return clip.with_effects([vfx.Resize(lambda t: 1 + 0.07 * (t / d))])


def build_clip(asset, w, h, fps, run_dir):
    dur = audio_duration(asset["audio"])
    if dur <= 0:
        dur = 3.0
    if asset.get("bg"):
        try:
            base = loop_to_clip(asset["bg"], dur)
            base = cover_crop(base, w, h)
            base = kenburns(base)
            ov = ImageClip(np.array(asset["overlay"])).with_duration(dur)
            return CompositeVideoClip([base, ov], size=(w, h)), dur
        except Exception as e:
            log("Footage clip error, using card: {}".format(e))
    return ImageClip(np.array(asset["card"])).with_duration(dur), dur


def assemble_video(assets, out, fps, w, h, fade, run_dir):
    auds = []
    try:
        for a in assets:
            auds.append(AudioFileClip(a["audio"]))
        full_audio = concatenate_audioclips(auds)

        clips = []
        for a in assets:
            c, dur = build_clip(a, w, h, fps, run_dir)
            clip = c.with_effects([vfx.CrossFadeIn(fade)]) if (fade > 0 and len(clips) > 0) else c
            clips.append(clip.with_duration(dur))

        t = 0.0
        layers = []
        for i, c in enumerate(clips):
            layers.append(c.with_start(t))
            t += c.duration - (fade if i > 0 else 0)

        final = CompositeVideoClip(layers, size=(w, h)).with_duration(t)
        final = final.with_audio(full_audio)
        final.write_videofile(
            out,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="2500k",
            preset="veryfast",
            threads=4,
            logger="bar",
        )
        log("Video saved: {} ({:.1f}s)".format(out, t))
        return out
    finally:
        try:
            full_audio.close()
        except Exception:
            pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            final.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# YouTube upload (YouTube Data API v3, free daily quota)
def get_service():
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, YT_SCOPES)
    except Exception:
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
        except Exception:
            return None
    if not creds or not creds.valid:
        return None
    return build("youtube", "v3", credentials=creds)


def auth_flow():
    if not os.path.exists(SECRET_PATH):
        print("client_secret.json not found in " + BASE_DIR)
        print()
        print("HOW TO GET IT (one-time, ~5 minutes):")
        print("  1. Go to https://console.cloud.google.com/apis/credentials")
        print("  2. Create project -> ENABLE API -> search 'YouTube Data API v3' -> enable")
        print("  3. Create credentials -> OAuth client ID -> Desktop app -> Download JSON")
        print("  4. Save the downloaded file as:  " + SECRET_PATH)
        print("  5. Re-run:  python youtube_agent.py --auth")
        return False
    flow = InstalledAppFlow.from_client_secrets_file(SECRET_PATH, YT_SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", open_browser=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())
    print("YouTube token saved. Uploads are now possible.")
    return True


def upload_video(service, path, title, description, tags, privacy):
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "tags": tags[:12],
            "categoryId": "27",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(path, chunksize=256 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    last = -1
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            if pct // 20 > last // 20:
                log("Upload {}%".format(pct))
                last = pct
    return response


# ---------------------------------------------------------------------------
# Report
def print_report():
    print("=== ZETTOOL YOUTUBE SHORTS STATUS ===")
    cfg = load_json(AGENT_CONFIG, required=False)
    up = (cfg.get("upload") or {}) if cfg else {}
    px = (cfg.get("pexels") or {}) if cfg else {}
    print("Auto-upload enabled: {}  (privacy: {})".format(up.get("enabled", False), up.get("privacy", "unlisted")))
    print("Pexels footage: {}  (key set: {})".format(px.get("enabled", False), bool(px.get("api_key"))))
    print("Token saved: {}".format(os.path.exists(TOKEN_PATH)))
    print("Client secret: {}".format(os.path.exists(SECRET_PATH)))
    st = load_state()
    today = dt.date.today().isoformat()
    cnt = st.get("daily_counts", {}).get(today, {})
    print("Today: {} generated / {} uploaded".format(cnt.get("generated", 0), cnt.get("uploaded", 0)))
    lim = (cfg or {}).get("daily_video_limit", 1)
    print("Daily video limit: {}".format(lim))
    print("Recent videos:")
    for v in st.get("videos", [])[-5:]:
        line = "  [{}] {}".format(v.get("date", ""), v.get("title", ""))
        if v.get("watch_url"):
            line += "  -> " + v["watch_url"]
        print(line)
    if not st.get("videos"):
        print("  (none yet)")
    print("Last run: " + st.get("last_run", ""))


# ---------------------------------------------------------------------------
def main():
    if "--report" in sys.argv:
        print_report()
        return

    log("=== ZetTool YouTube Agent starting ===")
    scfg = load_json(SITE_CONFIG)
    acfg = load_json(AGENT_CONFIG)
    if not acfg:
        log("ERROR: youtube_config.json missing or empty.")
        sys.exit(1)

    if "--auth" in sys.argv:
        auth_flow()
        return

    state = load_state()
    today = dt.date.today().isoformat()
    counts = state.setdefault("daily_counts", {}).setdefault(today, {"generated": 0, "uploaded": 0})
    lim = int(acfg.get("daily_video_limit", 1))
    force = "--force" in sys.argv

    if counts["generated"] >= lim and not force:
        log("Daily video limit reached ({}). Stopping.".format(lim))
        save_state(state)
        return

    dry_noupload = "--no-upload" in sys.argv
    up_cfg = acfg.get("upload") or {}
    privacy = up_cfg.get("privacy", "unlisted")
    px_cfg = acfg.get("pexels") or {}
    px_on = px_cfg.get("enabled", True) and bool(px_cfg.get("api_key"))
    fade = float(acfg.get("crossfade", 0.18))
    site_name = scfg.get("site_name", "ZetTool")

    topic = pick_topic(acfg, state, today)
    log("Topic: " + topic)
    script = build_script(scfg, acfg, topic)
    if not script:
        log("Could not build a script. Stopping.")
        return

    w = int(acfg.get("video_width", 1080))
    h = int(acfg.get("video_height", 1920))
    fps = int(acfg.get("fps", 30))
    voice = acfg.get("voice", "en-GB-RyanNeural")
    rate = acfg.get("voice_rate", "+8%")
    site = scfg.get("site") or "https://www.zettool.com"

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BUILD_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    segs = list(script["segments"])
    nseg = len(segs)
    cta_text = script.get("cta", "")
    assets = []

    for i, seg in enumerate(segs):
        label = "segment_{:02d}".format(i + 1)
        aud_path = os.path.join(run_dir, label + ".mp3")
        ok = tts_to_mp3(seg["text"], voice, rate, aud_path)
        dur = audio_duration(aud_path) if ok else 0
        if not ok or dur < 0.5:
            log("TTS failed/too short for {}".format(label))
            continue
        if px_on:
            bg = pexels_clip(px_cfg.get("api_key"), seg.get("visual", "money finance"), run_dir, i + 1, site_name)
        else:
            bg = None
        if bg:
            ov = render_segment_overlay(seg["text"], len(assets), nseg, w, h, is_hook=(i == 0))
            assets.append({"audio": aud_path, "bg": bg, "overlay": ov})
        else:
            assets.append({"audio": aud_path, "card": render_segment_card(seg["text"], len(assets), nseg, w, h, is_hook=(i == 0))})
        log("Voice OK: {} ({:.1f}s){}".format(label, dur, " + footage" if bg else " + card"))

    end_path = os.path.join(run_dir, "end.mp3")
    if cta_text and tts_to_mp3(cta_text, voice, rate, end_path) and audio_duration(end_path) > 1.0:
        assets.append({"audio": end_path, "card": render_end_card(w, h)})
        log("Voice OK: end ({:.1f}s)".format(audio_duration(end_path)))

    if len(assets) < 2:
        log("Not enough audio. Skipping run.")
        return

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]
    final_path = os.path.join(OUT_DIR, "{}_{}.mp4".format(today, slug))
    log("Assembling {} segments...".format(len(assets)))
    assemble_video(assets, final_path, fps, w, h, fade, run_dir)
    if not os.path.exists(final_path):
        log("Assembly failed.")
        return

    title = script["title"]
    suffix = (up_cfg.get("title_suffix", "#shorts") or "").strip()
    title_out = (title + " " + suffix).strip() if suffix else title
    desc = script.get("description") or ""
    desc += "\n\nFree tool: {}\nMore free tools: {}".format(script["tool_url"], site)
    tags = script.get("tags") or ["shorts", "calculator"]

    state_key = {"date": today, "topic": topic, "title": title_out, "path": final_path, "source": script.get("source", "?")}
    state.setdefault("used_topics", {})[topic] = today

    if dry_noupload or not up_cfg.get("enabled") or not os.path.exists(TOKEN_PATH):
        counts["generated"] += 1
        state_key["status"] = "generated"
        state.setdefault("videos", []).append(state_key)
        reason = "requested --no-upload" if dry_noupload else ("upload.disabled in config" if not up_cfg.get("enabled") else "token missing (run --auth)")
        log("Short READY: {}".format(final_path))
        log("Upload skipped: {}".format(reason))
        log("Title: {}".format(title_out))
    else:
        svc = get_service()
        if not svc:
            counts["generated"] += 1
            state_key["status"] = "generated"
            state.setdefault("videos", []).append(state_key)
            log("Short READY but token invalid. Run: python youtube_agent.py --auth")
        else:
            log("Uploading as {} ...".format(privacy))
            try:
                resp = upload_video(svc, final_path, title_out, desc, tags, privacy)
                vid = resp.get("id", "")
                counts["generated"] += 1
                counts["uploaded"] += 1
                state_key.update({
                    "status": "uploaded",
                    "video_id": vid,
                    "privacy": privacy,
                    "watch_url": "https://youtu.be/" + vid,
                })
                state.setdefault("videos", []).append(state_key)
                log("UPLOADED: https://youtu.be/{} (privacy {})".format(vid, privacy))
            except Exception as e:
                counts["generated"] += 1
                state_key["status"] = "generated"
                state.setdefault("videos", []).append(state_key)
                log("Upload failed: {}".format(e))

    save_state(state)
    log("Run complete. Today: {}/{} generated".format(counts["generated"], lim))


def tts_to_mp3(text, voice, rate, out):
    try:
        asyncio.run(_tts_save(text, voice, rate, out))
    except Exception as e:
        log("TTS error: {}".format(e))
        return False
    return os.path.exists(out) and os.path.getsize(out) > 800


async def _tts_save(text, voice, rate, out):
    com = edge_tts.Communicate(text, voice, rate=rate)
    await com.save(out)


if __name__ == "__main__":
    main()