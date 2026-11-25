"""Cinema Dark Mode UI components: CSS, Ambient Hero, Animated Popcorn Meter, Ticket-Stubs, XAI tokens."""
import hashlib
from html import escape
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import LABELS, MOVIE_PROFILES


def movie_accent(title: str) -> dict:
    if title in MOVIE_PROFILES:
        return MOVIE_PROFILES[title]
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()
    palettes = [
        ("#38BDF8", "#FACC15", "🎬"),
        ("#A855F7", "#FACC15", "📽️"),
        ("#F43F5E", "#38BDF8", "🍿"),
        ("#22D3EE", "#F59E0B", "🎞️"),
    ]
    p, s, motif = palettes[int(digest[:2], 16) % len(palettes)]
    return {
        "year": 2026,
        "genre": "Custom Audience Dataset",
        "accent_primary": p,
        "accent_secondary": s,
        "poster_motif": motif,
        "tagline": "Live ML, Aspect & RAG Audience Intelligence.",
    }


CINEMA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 12% 8%, rgba(56, 189, 248, 0.08), transparent 36%),
                radial-gradient(circle at 88% 12%, rgba(168, 85, 247, 0.09), transparent 38%),
                #0B0F19;
    color: #F8FAFC;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1224 0%, #151932 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.16);
}

/* Glassmorphic elevated cards */
.cinema-card {
    background: linear-gradient(145deg, rgba(21, 25, 50, 0.92), rgba(30, 27, 75, 0.72));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.55);
    margin-bottom: 14px;
}

.cinema-kpi {
    background: rgba(21, 25, 50, 0.85);
    border: 1px solid rgba(250, 204, 21, 0.22);
    border-radius: 14px;
    padding: 14px 16px;
    text-align: center;
}
.cinema-kpi .kpi-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
}
.cinema-kpi .kpi-val {
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: 4px;
}

/* Filmstrip decorative strip */
.filmstrip-bar {
    height: 12px;
    border-radius: 6px;
    background: repeating-linear-gradient(
        90deg,
        rgba(250, 204, 21, 0.75) 0px,
        rgba(250, 204, 21, 0.75) 14px,
        transparent 14px,
        transparent 24px
    );
    opacity: 0.45;
    margin: 6px 0 14px 0;
}

/* Ticket-Stub Summary Card */
.ticket-stub {
    position: relative;
    background: linear-gradient(135deg, #151932 0%, #1E1B4B 100%);
    border: 1px dashed rgba(250, 204, 21, 0.45);
    border-radius: 14px;
    padding: 15px 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.ticket-stub::before, .ticket-stub::after {
    content: "";
    position: absolute;
    top: 50%;
    width: 16px;
    height: 16px;
    background: #0B0F19;
    border-radius: 50%;
    transform: translateY(-50%);
}
.ticket-stub::before { left: -9px; border-right: 1px dashed rgba(250, 204, 21, 0.45); }
.ticket-stub::after { right: -9px; border-left: 1px dashed rgba(250, 204, 21, 0.45); }

.stamp {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    padding: 2px 9px;
    border-radius: 999px;
    text-transform: uppercase;
    margin-right: 6px;
}
.stamp-positive { background: rgba(250, 204, 21, 0.16); color: #FACC15; border: 1px solid rgba(250, 204, 21, 0.55); }
.stamp-neutral  { background: rgba(56, 189, 248, 0.16); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.55); }
.stamp-negative { background: rgba(244, 63, 94, 0.16); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.55); }
.stamp-irony    { background: rgba(168, 85, 247, 0.22); color: #D8B4FE; border: 1px solid rgba(168, 85, 247, 0.6); }
.stamp-spoiler  { background: rgba(249, 115, 22, 0.2); color: #FDBA74; border: 1px solid rgba(249, 115, 22, 0.6); }

/* Auto-Spoiler Shield */
details.spoiler-shield {
    margin-top: 6px;
    background: rgba(11, 15, 25, 0.65);
    border: 1px solid rgba(249, 115, 22, 0.35);
    border-radius: 10px;
    padding: 8px 12px;
}
details.spoiler-shield summary {
    cursor: pointer;
    color: #FDBA74;
    font-weight: 600;
    font-size: 0.85rem;
    list-style: none;
}
details.spoiler-shield:not([open]) .spoiler-preview {
    filter: blur(5px);
    user-select: none;
    opacity: 0.6;
}
.spoiler-revealed {
    margin-top: 6px;
    color: #E2E8F0;
    font-size: 0.94rem;
    line-height: 1.5;
}

/* Popcorn animations */
@keyframes popFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-5px) rotate(6deg); }
}
.pop-kernel {
    animation: popFloat 2.4s ease-in-out infinite;
    transform-origin: center;
}

/* Token XAI highlights */
.xai-tok {
    display: inline-block;
    padding: 2px 6px;
    margin: 2px 2px;
    border-radius: 6px;
    font-size: 0.96rem;
    line-height: 1.55;
}

/* Hide Streamlit Deploy button */
[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
.stAppDeployButton {
    display: none !important;
}
</style>
"""


def hero_header_html(title: str, n_reviews: int, rag_backend: str) -> str:
    prof = movie_accent(title)
    p, s = prof["accent_primary"], prof["accent_secondary"]
    return f"""
    <div style="
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 16px;
        background:
            radial-gradient(circle at 18% 30%, {p}38, transparent 52%),
            radial-gradient(circle at 86% 25%, {s}30, transparent 48%),
            linear-gradient(135deg, #151932 0%, #0B0F19 100%);
        border: 1px solid {p}44;
        box-shadow: 0 18px 42px rgba(2, 6, 23, 0.65);
    ">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:18px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:18px;">
          <div style="
              width:70px; height:94px; border-radius:12px;
              background: linear-gradient(160deg, {p}55, #1E1B4B);
              border: 1px solid {s}88;
              display:flex; align-items:center; justify-content:center;
              font-size:2.2rem; box-shadow: 0 0 28px {p}44;
          ">{prof['poster_motif']}</div>
          <div>
            <div style="font-size:0.74rem; letter-spacing:0.14em; text-transform:uppercase; color:{s}; font-weight:700;">
              🎬 SLAYORSKIP STUDIO &bull; {escape(prof['genre'])} &bull; {prof['year']}
            </div>
            <h1 style="margin:4px 0 4px 0; font-size:2.05rem; font-weight:800; color:#F8FAFC;">
              {escape(title)}
            </h1>
            <div style="color:#CBD5E1; font-size:0.92rem;">{escape(prof['tagline'])}</div>
          </div>
        </div>
        <div style="text-align:right;">
          <span class="stamp stamp-neutral">{n_reviews} REVIEWS INDEXED</span><br/>
          <span style="font-size:0.75rem; color:#94A3B8;">RAG Vector Engine: {escape(rag_backend)}</span>
        </div>
      </div>
      <div class="filmstrip-bar"></div>
    </div>
    """


def popcorn_meter_html(popcorn: dict) -> str:
    score = popcorn["score"]
    fill_h = max(6, min(92, int(0.92 * score)))
    fill_y = 112 - fill_h
    color = popcorn["color"]
    kernels = ""
    if score >= 30:
        kernels += '<circle class="pop-kernel" cx="48" cy="24" r="9" fill="#FEF08A"/>'
        kernels += '<circle class="pop-kernel" cx="62" cy="18" r="10" fill="#FACC15"/>'
        kernels += '<circle class="pop-kernel" cx="76" cy="25" r="8" fill="#FDE047"/>'
    if score >= 60:
        kernels += '<circle class="pop-kernel" cx="38" cy="30" r="7" fill="#FDE047"/>'
        kernels += '<circle class="pop-kernel" cx="84" cy="30" r="7" fill="#FEF08A"/>'
    return f"""
    <div class="cinema-card" style="display:flex; align-items:center; gap:20px; border-color:{color}66;">
      <svg width="110" height="122" viewBox="0 0 120 130" style="flex-shrink:0;">
        <defs>
          <clipPath id="bucketClip">
            <polygon points="24,32 96,32 84,118 36,118"/>
          </clipPath>
        </defs>
        {kernels}
        <polygon points="24,32 96,32 84,118 36,118" fill="#0B0F19" stroke="{color}" stroke-width="2.5"/>
        <rect x="20" y="{fill_y}" width="80" height="{fill_h + 12}" fill="{color}" opacity="0.86" clip-path="url(#bucketClip)"/>
        <line x1="44" y1="34" x2="49" y2="116" stroke="#0B0F19" stroke-width="3" opacity="0.45"/>
        <line x1="60" y1="34" x2="60" y2="116" stroke="#0B0F19" stroke-width="3" opacity="0.45"/>
        <line x1="76" y1="34" x2="71" y2="116" stroke="#0B0F19" stroke-width="3" opacity="0.45"/>
      </svg>
      <div>
        <div style="font-size:0.75rem; letter-spacing:0.12em; text-transform:uppercase; color:#94A3B8; font-weight:700;">
          🍿 Animated Popcorn Meter
        </div>
        <div style="font-size:2.3rem; font-weight:800; color:{color}; line-height:1.1; margin:4px 0;">
          {score:.0f}% <span style="font-size:1.05rem; font-weight:700; color:#F8FAFC;">{escape(popcorn['tier'])}</span>
        </div>
        <div style="color:#CBD5E1; font-size:0.88rem;">{escape(popcorn['subtitle'])}</div>
      </div>
    </div>
    """


def ticket_stub_html(text: str, label: str, extra_badges: List[str] = None,
                     has_spoiler: bool = False, meta_right: str = "") -> str:
    badges = f'<span class="stamp stamp-{escape(label)}">{escape(label)}</span>'
    for b in extra_badges or []:
        if "SPOILER" in b.upper():
            cls = "stamp-spoiler"
        elif "IRONY" in b.upper():
            cls = "stamp-irony"
        elif "|" in b:
            cls = "stamp-neutral"
        else:
            cls = "stamp-irony"
        badges += f'<span class="stamp {cls}">{escape(b)}</span>'

    safe = escape(text)
    if has_spoiler:
        body = f"""
        <details class="spoiler-shield">
          <summary>⚠️ Auto-Spoiler Shield Active — Click to Reveal Plot Spoiler</summary>
          <div class="spoiler-revealed">{safe}</div>
        </details>
        """
    else:
        body = f'<div style="margin-top:8px; color:#E2E8F0; font-size:0.95rem; line-height:1.5;">{safe}</div>'

    right_html = f'<span style="font-size:0.76rem; color:#94A3B8;">{escape(meta_right)}</span>' if meta_right else ""
    return f"""
    <div class="ticket-stub">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
        <div>{badges}</div>
        {right_html}
      </div>
      {body}
    </div>
    """


def xai_highlighted_html(tokens: List[dict]) -> str:
    spans = []
    for item in tokens:
        tok = escape(item["token"])
        w = item["weight"]
        if w > 0.08:
            alpha = min(0.55, 0.16 + 0.4 * abs(w))
            style = f"background:rgba(250, 204, 21, {alpha:.2f}); border:1px solid rgba(74, 222, 128, 0.65); color:#FEF9C3;"
        elif w < -0.08:
            alpha = min(0.55, 0.16 + 0.4 * abs(w))
            style = f"background:rgba(244, 63, 94, {alpha:.2f}); border:1px solid rgba(251, 113, 133, 0.7); color:#FFE4E6;"
        else:
            style = "background:rgba(148, 163, 184, 0.08); color:#CBD5E1;"
        spans.append(f'<span class="xai-tok" title="weight: {w:+.2f}" style="{style}">{tok}</span>')
    return f'<div class="cinema-card" style="line-height:2.0;">{" ".join(spans)}</div>'


def plot_roc_curves(roc_curves: dict, model_name: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor="#0B0F19")
    ax.set_facecolor("#151932")
    palette = {"negative": "#F43F5E", "neutral": "#38BDF8", "positive": "#FACC15"}
    for label in LABELS:
        c = roc_curves[label]
        ax.plot(c["fpr"], c["tpr"], color=palette[label], linewidth=2.2,
                label=f"{label.capitalize()} (AUC = {c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#64748B", linewidth=1.1)
    ax.set_title(f"One-vs-Rest ROC Curves — {model_name}", color="#F8FAFC", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("False Positive Rate", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("True Positive Rate", color="#CBD5E1", fontsize=9)
    ax.tick_params(colors="#CBD5E1", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    leg = ax.legend(fontsize=8, facecolor="#0B0F19", edgecolor="#334155")
    for t in leg.get_texts():
        t.set_color("#E2E8F0")
    fig.tight_layout()
    return fig


def plot_battle_radar(title_a: str, emo_a: Dict[str, float], title_b: str, emo_b: Dict[str, float]):
    labels = list(emo_a.keys())
    vals_a = [emo_a[k] for k in labels]
    vals_b = [emo_b[k] for k in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_loop = angles + angles[:1]

    fig = plt.figure(figsize=(5.2, 4.5), facecolor="#0B0F19")
    ax = fig.add_subplot(111, polar=True, facecolor="#151932")
    ax.plot(angles_loop, vals_a + vals_a[:1], color="#FACC15", linewidth=2.3, label=title_a)
    ax.fill(angles_loop, vals_a + vals_a[:1], color="#FACC15", alpha=0.22)
    ax.plot(angles_loop, vals_b + vals_b[:1], color="#38BDF8", linewidth=2.3, label=title_b)
    ax.fill(angles_loop, vals_b + vals_b[:1], color="#38BDF8", alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#F8FAFC", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#94A3B8", fontsize=7)
    ax.grid(color="#334155", alpha=0.7)
    ax.spines["polar"].set_color("#475569")
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), fontsize=8,
                    facecolor="#0B0F19", edgecolor="#334155")
    for t in leg.get_texts():
        t.set_color("#E2E8F0")
    fig.tight_layout()
    return fig
