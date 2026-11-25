"""SlayOrSkip — Movie Review Intelligence & Sentiment Studio (Cinema Dark Mode)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.aspects import ASPECT_LABELS
from src.config import DATA_DIR, REPORTS_DIR
from src.data import load_reviews_file
from src.predict import SentimentModel
from src.report import (analyze_movie, answer_global_question,
                        build_global_rag_bundle, build_report, export_docx,
                        export_pdf, slugify)
from src.ui import (CINEMA_CSS, hero_header_html, plot_battle_radar,
                    popcorn_meter_html, ticket_stub_html)

st.set_page_config(
    page_title="SlayOrSkip • Sentiment Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CINEMA_CSS, unsafe_allow_html=True)

SAMPLE_CSV = DATA_DIR / "sample_reviews.csv"


@st.cache_resource(show_spinner="Loading SlayOrSkip ML engine…")
def get_sentiment_model(prefer: str) -> SentimentModel:
    return SentimentModel(prefer)


@st.cache_data
def load_sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@st.cache_resource(show_spinner="Indexing global 6-genre RAG database…")
def get_global_rag_bundle(prefer: str, adjust_sarcasm: bool) -> dict:
    df_all = load_sample_df()
    model = get_sentiment_model(prefer)
    return build_global_rag_bundle(df_all, model, adjust_sarcasm=adjust_sarcasm)


def _run_analysis(movie_title: str, df_reviews: pd.DataFrame, model: SentimentModel,
                  use_llm: bool, adjust_sarcasm: bool) -> dict:
    analysis = analyze_movie(movie_title, df_reviews, model, adjust_sarcasm=adjust_sarcasm)
    return build_report(analysis, use_llm=use_llm)


if "global_chat" not in st.session_state:
    st.session_state.global_chat = []


# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown("## 🎬 SlayOrSkip")
    st.caption("Cinema Dark Mode • ML + RAG Studio")

    nav = st.radio(
        "Studio Navigation",
        ["Dashboard", "Aspect Lab", "Ask the Audience", "Head-to-Head", "PDF Dossier"],
        label_visibility="collapsed",
    )
    st.divider()

    st.subheader("Data Ingestion")
    src_choice = st.radio("Source", ["Curated Movies", "Upload CSV / JSON", "Paste Reviews"], horizontal=False)

    sample_df = load_sample_df()
    reviews_df, movie_title = None, "Neon Horizon"

    if src_choice == "Curated Movies":
        genre_map = (
            sample_df.dropna(subset=["movie"])
            .drop_duplicates(subset=["movie"])
            .set_index("movie")["genre"]
            .to_dict()
            if "genre" in sample_df.columns
            else {}
        )
        movies_avail = list(genre_map.keys()) if genre_map else sorted(sample_df["movie"].dropna().unique())
        movie_title = st.selectbox(
            "Select Film (Genre — Movie)",
            movies_avail,
            format_func=lambda m: f"{genre_map.get(m, 'Movie')} — {m}",
        )
        reviews_df = sample_df[sample_df["movie"] == movie_title].copy()
    elif src_choice == "Upload CSV / JSON":
        up = st.file_uploader("Upload .csv or .json ('text' column required)", type=["csv", "json"])
        available_movies = []
        if up is not None:
            loaded = load_reviews_file(up, filename=up.name)
            if "movie" in loaded.columns and loaded["movie"].notna().any():
                movies_in_file = sorted(loaded["movie"].dropna().unique())
                movie_title = st.selectbox("Film in file", movies_in_file)
                reviews_df = loaded[loaded["movie"] == movie_title].copy()
                available_movies = movies_in_file
            else:
                movie_title = st.text_input("Movie Title", value="Custom Upload")
                reviews_df = loaded
                available_movies = [movie_title]
    else:
        movie_title = st.text_input("Movie Title", value="Custom Screening")
        raw_txt = st.text_area("Paste reviews (one per line)", height=130,
                               placeholder="The cinematography is stunning!\nA masterpiece if you're trying to cure insomnia.")
        lines = [l.strip() for l in raw_txt.splitlines() if l.strip()]
        if lines:
            reviews_df = pd.DataFrame({"text": lines})
        available_movies = [movie_title]

    st.divider()
    st.subheader("Engine Settings")
    prefer = st.selectbox("Primary Sentiment Engine", ["auto", "baseline", "bert"],
                          help="auto uses DistilBERT if available & supported, otherwise best SST-5 baseline.")
    adjust_sarcasm = st.toggle("Sarcasm & Irony Radar Adjustment", value=True)
    use_llm = st.checkbox("LLM Narrative Synthesis (ANTHROPIC_API_KEY)", value=False)

try:
    primary_model = get_sentiment_model(prefer)
    st.sidebar.success(f"Engine Ready: {primary_model.kind.upper()}")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Automatically compute/refresh active movie analysis when selection changes (never clears global_chat)
if reviews_df is not None and not reviews_df.empty:
    cache_sig = (movie_title, len(reviews_df), reviews_df["text"].iloc[0], adjust_sarcasm, use_llm, primary_model.kind)
    if st.session_state.get("active_sig") != cache_sig:
        with st.spinner("🎬 Rolling film & running ML + FAISS RAG pipeline…"):
            st.session_state["report"] = _run_analysis(movie_title, reviews_df, primary_model, use_llm, adjust_sarcasm)
            st.session_state["active_sig"] = cache_sig

rep = st.session_state.get("report")
global_bundle = get_global_rag_bundle(prefer, adjust_sarcasm)

# ------------------------------------------------------------------ Views
if nav == "Dashboard":
    if not rep:
        st.info("Upload a CSV/JSON file or paste reviews in the sidebar to launch analysis.")
        st.stop()

    st.markdown(hero_header_html(rep["title"], rep["n_reviews"], rep["rag_backend"]), unsafe_allow_html=True)

    c_pop, c_stats = st.columns([1.2, 1.8])
    with c_pop:
        st.markdown(popcorn_meter_html(rep["popcorn"]), unsafe_allow_html=True)
    with c_stats:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="cinema-kpi"><div class="kpi-label">Positive</div><div class="kpi-val" style="color:#FACC15;">{rep["pct"]["positive"]:.0f}%</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="cinema-kpi"><div class="kpi-label">Neutral</div><div class="kpi-val" style="color:#38BDF8;">{rep["pct"]["neutral"]:.0f}%</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="cinema-kpi"><div class="kpi-label">Negative</div><div class="kpi-val" style="color:#FB7185;">{rep["pct"]["negative"]:.0f}%</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="cinema-kpi"><div class="kpi-label">Irony / Spoilers</div><div class="kpi-val" style="color:#D8B4FE;">{rep["sarcasm_count"]} / {rep["spoiler_count"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cinema-card" style="margin-top:12px;"><b>Executive Consensus</b><br/><span style="color:#CBD5E1;">{rep["overview"]}</span></div>', unsafe_allow_html=True)

    col_chart, col_radar = st.columns([1.65, 1.0])
    with col_chart:
        st.image(rep["chart_path"], use_container_width=True)
    with col_radar:
        st.image(rep["radar_path"], use_container_width=True)

    st.markdown("### 🎟️ Ticket-Stub Verdict & Top RAG Evidence")
    st.markdown(
        ticket_stub_html(
            rep["verdict"],
            "positive" if rep["popcorn"]["score"] >= 60 else ("neutral" if rep["popcorn"]["score"] >= 45 else "negative"),
            extra_badges=[rep["popcorn"]["stamp"], f"{rep['generator']}"],
            meta_right=f"{rep['n_reviews']} reviews analyzed",
        ),
        unsafe_allow_html=True,
    )

    ev_l, ev_r = st.columns(2)
    with ev_l:
        st.markdown("#### 🌟 Top Positive Citations")
        for h in rep["evidence"]["positive"]:
            badges = []
            if h.get("is_sarcastic"):
                badges.append("IRONY RADAR")
            if h.get("has_spoiler"):
                badges.append("SPOILER")
            st.markdown(
                ticket_stub_html(h["text"], "positive", extra_badges=badges,
                                 has_spoiler=bool(h.get("has_spoiler")),
                                 meta_right=f"cos={h['similarity']:.2f}"),
                unsafe_allow_html=True,
            )
    with ev_r:
        st.markdown("#### 🎭 Top Critical Citations")
        for h in rep["evidence"]["negative"]:
            badges = []
            if h.get("is_sarcastic"):
                badges.append("IRONY RADAR")
            if h.get("has_spoiler"):
                badges.append("SPOILER")
            st.markdown(
                ticket_stub_html(h["text"], "negative", extra_badges=badges,
                                 has_spoiler=bool(h.get("has_spoiler")),
                                 meta_right=f"cos={h['similarity']:.2f}"),
                unsafe_allow_html=True,
            )

    st.markdown("### 🧪 Preprocessing Stats & Sentiment Drift")
    cs = rep["corpus_stats"]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Tokens", cs["total_tokens"])
    p2.metric("Stopwords Removed", f"{cs['stopwords_removed']} ({cs['stopword_pct']}%)")
    p3.metric("Lemmatized Vocab", cs["vocab_size"])
    p4.metric("Avg Tokens / Review", cs["avg_tokens"])

    df_rev = rep["reviews"].copy()
    if len(df_rev) >= 2:
        drift_x = df_rev["date"] if "date" in df_rev.columns else [f"#{i+1}" for i in range(len(df_rev))]
        fig_d, ax_d = plt.subplots(figsize=(9.5, 2.5), facecolor="#0B0F19")
        ax_d.set_facecolor("#151932")
        rolling = df_rev["sentiment_score"].rolling(window=min(4, len(df_rev)), min_periods=1).mean()
        ax_d.plot(range(len(df_rev)), df_rev["sentiment_score"], "o", color="#38BDF8", alpha=0.55, label="Review Polarity")
        ax_d.plot(range(len(df_rev)), rolling, color="#FACC15", linewidth=2.4, label="Rolling Sentiment Drift")
        ax_d.axhline(0, color="#64748B", linestyle="--", linewidth=0.9)
        ax_d.set_ylim(-1.25, 1.25)
        ax_d.set_yticks([-1, 0, 1])
        ax_d.set_yticklabels(["Negative", "Neutral", "Positive"], color="#E2E8F0", fontsize=8)
        step = max(1, len(df_rev) // 7)
        ax_d.set_xticks(range(0, len(df_rev), step))
        ax_d.set_xticklabels([str(list(drift_x)[i]) for i in range(0, len(df_rev), step)], color="#CBD5E1", fontsize=8)
        for sp in ax_d.spines.values():
            sp.set_color("#334155")
        leg = ax_d.legend(fontsize=8, facecolor="#0B0F19", edgecolor="#334155")
        for t in leg.get_texts():
            t.set_color("#E2E8F0")
        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

elif nav == "Aspect Lab":
    if not rep:
        st.info("Load reviews first.")
        st.stop()

    st.markdown(f"## 🎞️ Aspect Lab & Emotion Spectrum — {rep['title']}")
    c_tbl, c_rad = st.columns([1.45, 1.0])
    with c_tbl:
        disp = rep["aspect_summary"][["aspect_name", "positive", "neutral", "negative", "mentions", "net_score"]].rename(
            columns={"aspect_name": "Cinematic Dimension", "net_score": "Net Score"}
        )
        st.dataframe(disp, hide_index=True, use_container_width=True)
    with c_rad:
        st.image(rep["radar_path"], use_container_width=True)

    st.markdown("### 🔍 RAG Evidence by Cinematic Dimension")
    asp_choice = st.selectbox("Select Aspect", list(ASPECT_LABELS.keys()), format_func=lambda a: ASPECT_LABELS[a])
    ev = rep["evidence"]["aspects"][asp_choice]
    al, ar = st.columns(2)
    with al:
        st.markdown(f"#### Praise for {ASPECT_LABELS[asp_choice]}")
        if ev["positive"]:
            for h in ev["positive"]:
                st.markdown(ticket_stub_html(h["text"], "positive", has_spoiler=bool(h.get("has_spoiler")), meta_right=f"cos={h['similarity']:.2f}"), unsafe_allow_html=True)
        else:
            st.caption("No positive aspect quotes retrieved.")
    with ar:
        st.markdown(f"#### Criticism of {ASPECT_LABELS[asp_choice]}")
        if ev["negative"]:
            for h in ev["negative"]:
                st.markdown(ticket_stub_html(h["text"], "negative", has_spoiler=bool(h.get("has_spoiler")), meta_right=f"cos={h['similarity']:.2f}"), unsafe_allow_html=True)
        else:
            st.caption("No negative aspect quotes retrieved.")

    st.markdown("### 🛡️ Sarcasm Radar & Spoiler Audit Log")
    audit_cols = ["text", "raw_label", "label", "is_sarcastic", "sarcasm_reason", "has_spoiler", "spoiler_triggers"]
    st.dataframe(rep["reviews"][audit_cols], use_container_width=True)

elif nav == "Ask the Audience":
    hdr_l, hdr_r = st.columns([4, 1])
    with hdr_l:
        st.markdown("## 💬 Ask the Audience — Global 6-Genre RAG Chat")
        st.caption(
            f"Decoupled from sidebar selection • Searches all 6 curated movies ({global_bundle['rag_backend']}) "
            "with automatic Movie & Genre detection."
        )
    with hdr_r:
        st.write("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.global_chat = []
            st.rerun()

    sugg_cols = st.columns(3)
    suggestions = [
        "How was the acting in Shadow Protocol?",
        "How is the Comedy movie?",
        "Which movie has the best reviews overall?",
    ]
    clicked_q = None
    for col, q in zip(sugg_cols, suggestions):
        if col.button(q, use_container_width=True):
            clicked_q = q

    user_q = st.chat_input("Ask about any movie, genre (Sci-Fi, Action, Horror, Romance, Comedy, Mystery / Drama), or chat…") or clicked_q
    if user_q:
        res = answer_global_question(
            user_q,
            global_bundle,
            use_llm=use_llm,
            chat_history=st.session_state.global_chat,
        )
        st.session_state.global_chat.append({"q": user_q, **res})

    for item in st.session_state.global_chat:
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            scope_txt = item.get("scope", "All Movies (6 Genres)")
            st.caption(f"🎯 Scope: {scope_txt} • Source: {item['generator']}")
            cites = item.get("citations") or []
            if cites:
                with st.expander(f"View {len(cites)} Cited Audience Reviews"):
                    for h in cites:
                        movie_tag = f"{h.get('movie', 'Movie')} | {h.get('genre', 'Genre')}"
                        badges = [movie_tag]
                        if h.get("is_sarcastic"):
                            badges.append("IRONY RADAR")
                        if h.get("has_spoiler"):
                            badges.append("SPOILER")
                        st.markdown(
                            ticket_stub_html(
                                h["text"],
                                h["label"],
                                extra_badges=badges,
                                has_spoiler=bool(h.get("has_spoiler")),
                                meta_right=f"cos={h['similarity']:.2f}",
                            ),
                            unsafe_allow_html=True,
                        )

elif nav == "Head-to-Head":
    st.markdown("## ⚔️ Movie vs. Movie Battle Mode")
    all_movies = sorted(sample_df["movie"].dropna().unique())
    if len(all_movies) < 2:
        st.warning("Need at least two movies in the sample dataset for Battle Mode.")
        st.stop()

    b1, b2 = st.columns(2)
    film_a = b1.selectbox("Challenger A", all_movies, index=0)
    film_b = b2.selectbox("Challenger B", all_movies, index=min(1, len(all_movies) - 1))

    with st.spinner("Computing side-by-side Popcorn Meters & aspect battle…"):
        ana_a = analyze_movie(film_a, sample_df[sample_df["movie"] == film_a], primary_model, adjust_sarcasm=adjust_sarcasm)
        ana_b = analyze_movie(film_b, sample_df[sample_df["movie"] == film_b], primary_model, adjust_sarcasm=adjust_sarcasm)

    winner = film_a if ana_a["popcorn"]["score"] >= ana_b["popcorn"]["score"] else film_b
    st.markdown(
        f'<div class="cinema-card" style="text-align:center; border-color:#FACC15;">'
        f'<span class="stamp stamp-positive">👑 BOX OFFICE CROWD WINNER</span> '
        f'<b style="font-size:1.25rem; color:#FACC15;">{winner}</b> '
        f'({ana_a["popcorn"]["score"]:.0f}% vs {ana_b["popcorn"]["score"]:.0f}% Popcorn Meter)</div>',
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"### {film_a}")
        st.markdown(popcorn_meter_html(ana_a["popcorn"]), unsafe_allow_html=True)
    with m2:
        st.markdown(f"### {film_b}")
        st.markdown(popcorn_meter_html(ana_b["popcorn"]), unsafe_allow_html=True)

    c_asp, c_emo = st.columns([1.2, 1.0])
    with c_asp:
        st.markdown("### Aspect Net Score Comparison")
        sa = ana_a["aspect_summary"][["aspect_name", "net_score"]].rename(columns={"net_score": film_a})
        sb = ana_b["aspect_summary"][["aspect_name", "net_score"]].rename(columns={"net_score": film_b})
        merged = sa.merge(sb, on="aspect_name")
        st.dataframe(merged.rename(columns={"aspect_name": "Cinematic Dimension"}), hide_index=True, use_container_width=True)
    with c_emo:
        st.markdown("### Emotion Spectrum Overlay")
        fig_b = plot_battle_radar(film_a, ana_a["emotions"], film_b, ana_b["emotions"])
        st.pyplot(fig_b)
        plt.close(fig_b)

elif nav == "PDF Dossier":
    if not rep:
        st.info("Load reviews first.")
        st.stop()

    st.markdown(f"## 📄 Box Office Intelligence Dossier — {rep['title']}")
    st.markdown(
        f'<div class="cinema-card">'
        f'<span class="stamp stamp-positive">POPCORN METER: {rep["popcorn"]["score"]:.0f}% ({rep["popcorn"]["tier"]})</span>'
        f'<span class="stamp stamp-neutral">{rep["n_reviews"]} REVIEWS</span>'
        f'<span class="stamp stamp-irony">{rep["generator"]}</span>'
        f'<h4 style="margin-top:12px;">Executive Consensus</h4><p style="color:#CBD5E1;">{rep["overview"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    l, r = st.columns(2)
    with l:
        st.markdown("### ✅ Strengths")
        for x in rep["strengths"]:
            st.markdown(f"- {x}")
    with r:
        st.markdown("### ⚠️ Weaknesses")
        for x in rep["weaknesses"]:
            st.markdown(f"- {x}")

    if rep.get("takeaways"):
        st.markdown("### 💡 Audience Takeaways")
        for x in rep["takeaways"]:
            st.markdown(f"- {x}")

    st.markdown("### 🏆 Final Verdict")
    st.info(rep["verdict"])

    REPORTS_DIR.mkdir(exist_ok=True)
    slug = slugify(rep["title"])
    pdf_path = export_pdf(rep, REPORTS_DIR / f"{slug}_dossier.pdf")
    docx_path = export_docx(rep, REPORTS_DIR / f"{slug}_dossier.docx")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download Cinema PDF Dossier", Path(pdf_path).read_bytes(), f"{slug}_dossier.pdf", "application/pdf", use_container_width=True)
    d2.download_button("⬇️ Download DOCX Report", Path(docx_path).read_bytes(), f"{slug}_dossier.docx", use_container_width=True)
# Testing Jenkins Auto-Trigger CI/CD
# Auto trigger test 