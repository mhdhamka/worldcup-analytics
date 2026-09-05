import sys
import os
import string

import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
from src.match_engine.predictor import WorldCupMatchPredictor
from src.player_clustering.clusterer import PlayerStyleClusterer, SAMPLE_PLAYERS
from teams_data import team_list, get_flag, seeded_groups

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="FIFA World Cup 2026 | Analytics Hub", page_icon="⚽", layout="wide")

# ============================================================
# GLOBAL STYLING -- official-tournament look: deep pitch green,
# gold trophy accents, crest-style badges
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 50% -10%, #0f3d2e 0%, #08251b 40%, #04140f 100%);
    color: #f2f6f4;
}
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1600px; }

.wc-header {
    position: relative; overflow: hidden;
    background:
        radial-gradient(circle at 85% 15%, rgba(212,175,55,0.18), transparent 40%),
        linear-gradient(135deg, #062017, #0d3324 55%, #052015);
    border: 1px solid rgba(212,175,55,0.25);
    border-bottom: 3px solid #d4af37;
    padding: 30px 36px; border-radius: 18px; margin-bottom: 16px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.55), inset 0 1px rgba(255,255,255,0.04);
}
.wc-kicker {
    color: #d4af37; font-size: 0.78rem; font-weight: 800;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 6px;
}
.wc-title {
    font-size: 2.6rem; font-weight: 900; letter-spacing: -0.8px; margin: 0;
    background: linear-gradient(90deg, #ffffff, #b7f7d8 40%, #d4af37);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.wc-subtitle { color: #9fe6c4; font-size: 0.9rem; font-weight: 700; margin-top: 10px; }

.disclaimer-box {
    background: rgba(212,175,55,0.08); border: 1px solid rgba(212,175,55,0.28);
    border-radius: 10px; padding: 10px 18px; font-size: 0.82rem; color: #d8e6dd; margin-bottom: 20px;
}

div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0f2e21, #081b13);
    border: 1px solid #1f4a34; padding: 16px; border-radius: 12px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"]:hover { border-color: #d4af37; }

.stButton > button {
    border-radius: 10px; font-weight: 800; letter-spacing: 0.3px;
    border: 1px solid rgba(212,175,55,0.4); transition: all 0.2s ease;
    background: linear-gradient(135deg, #14543a, #0c3826);
}
.stButton > button:hover { transform: translateY(-1px); border-color: #d4af37; box-shadow: 0 8px 25px rgba(212,175,55,0.2); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #d4af37, #b8912a); color: #082015; border: none;
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 8px 25px rgba(212,175,55,0.35); }

button[data-baseweb="tab"] { font-weight: 800; letter-spacing: 0.3px; }
h1, h2, h3 { letter-spacing: -0.3px; }

.group-card {
    background: linear-gradient(145deg, #0f2e21, #081b13);
    border: 1px solid #1f4a34; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
}
.group-card h5 { color: #d4af37; margin: 0 0 8px 0; letter-spacing: 1px; text-transform: uppercase; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="wc-header">
    <div class="wc-kicker">Fan-Made Data Science Project</div>
    <div class="wc-title">⚽ FIFA World Cup 2026 Analytics Hub</div>
    <div class="wc-subtitle">48-Team Group Draw · Match Prediction · Player Style Clustering · Knockout Simulator</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
ℹ️ Data-science demo project. Team stats, the group draw, and match history are illustrative
synthetic data (not a live feed) and every score/bracket is a <b>model projection</b>, not a real result.
</div>
""", unsafe_allow_html=True)

# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner="Training match prediction model on 48 teams...")
def get_predictor():
    return WorldCupMatchPredictor()


@st.cache_data(show_spinner="Clustering player styles...")
def get_clustered_players():
    clusterer = PlayerStyleClusterer(n_clusters=4)
    df = clusterer.cluster_players(SAMPLE_PLAYERS)
    return df, clusterer.cluster_labels


@st.cache_data(show_spinner="Drawing groups...")
def get_groups():
    return seeded_groups(n_groups=12, group_size=4)


predictor = get_predictor()
teams_list = team_list()
groups = get_groups()


def team_label(team: str) -> str:
    return f"{get_flag(team)} {team}"


with st.sidebar:
    st.markdown("### Model transparency")
    st.metric("Hold-out test accuracy", f"{predictor.test_accuracy}%")
    st.caption(
        "Trained on synthetic, strength-weighted match data (see predictor.py). "
        "Baseline random-guess accuracy for a 3-way outcome is ~33%."
    )
    st.markdown("---")
    st.caption(f"{len(teams_list)} teams · 12 groups of 4 · Round of 32 knockout")

# ============================================================
# BRACKET RENDERER (Round of 16 -> Final, Champions-League style)
# ============================================================

def render_champions_league_bracket(bracket_results):
    def get_team(match, side):
        prediction = match.get("prediction", {})
        if side == "home":
            return prediction.get("home_team") or match["match"].split(" vs ")[0]
        return prediction.get("away_team") or match["match"].split(" vs ")[-1]

    def get_probability(match, side):
        prediction = match.get("prediction", {})
        key = "home_win_prob" if side == "home" else "away_win_prob"
        return float(prediction.get(key, 0))

    def team_row(match, side):
        team = get_team(match, side)
        probability = get_probability(match, side)
        winner = match.get("winner") == team
        flag = get_flag(team)
        return f"""
        <div class="team-row {'winner' if winner else ''}">
            <div class="team-left">
                <div class="team-badge">{flag}</div>
                <span class="team-name">{team}</span>
            </div>
            <div class="team-prob">{probability:.0f}%</div>
            {'<div class="winner-mark">✓</div>' if winner else ''}
        </div>
        """

    def match_card(match, round_name, index):
        prediction = match.get("prediction", {})
        home_team, away_team = get_team(match, "home"), get_team(match, "away")
        home_p, away_p = get_probability(match, "home"), get_probability(match, "away")
        winner = match.get("winner", "")
        xg_home, xg_away = prediction.get("home_xg", "-"), prediction.get("away_xg", "-")
        card_id = f"{round_name.lower().replace(' ', '-')}-{index}"
        return f"""
        <div class="match-card" id="{card_id}" data-home="{home_team}" data-away="{away_team}">
            <div class="match-header"><span>{round_name}</span><span>#{index + 1}</span></div>
            {team_row(match, "home")}
            {team_row(match, "away")}
            <div class="probability-container">
                <div class="probability-bar"><div class="probability-home" style="width:{home_p}%"></div></div>
                <div class="probability-labels"><span>{home_p:.0f}%</span><span>{away_p:.0f}%</span></div>
            </div>
            <div class="match-footer"><span>xG {xg_home} - {xg_away}</span><span class="predicted">{winner}</span></div>
        </div>
        """

    r16 = bracket_results.get("Round of 16", [])
    qf = bracket_results.get("Quarter-Finals", [])
    sf = bracket_results.get("Semi-Finals", [])
    final = bracket_results.get("Final")

    left_r16, right_r16 = r16[:4], r16[4:]
    left_qf, right_qf = qf[:2], qf[2:]

    left_r16_html = "".join(match_card(m, "Round of 16", i) for i, m in enumerate(left_r16))
    right_r16_html = "".join(match_card(m, "Round of 16", i + 4) for i, m in enumerate(right_r16))
    left_qf_html = "".join(match_card(m, "Quarter-Final", i) for i, m in enumerate(left_qf))
    right_qf_html = "".join(match_card(m, "Quarter-Final", i + 2) for i, m in enumerate(right_qf))
    left_sf_html = match_card(sf[0], "Semi-Final", 0) if len(sf) >= 1 else ""
    right_sf_html = match_card(sf[1], "Semi-Final", 1) if len(sf) >= 2 else ""

    final_html = ""
    if final:
        prediction = final.get("prediction", {})
        home_team, away_team = get_team(final, "home"), get_team(final, "away")
        home_p, away_p = get_probability(final, "home"), get_probability(final, "away")
        champion = final.get("winner", bracket_results.get("Champion", ""))
        final_html = f"""
        <div class="final-card">
            <div class="final-header"><span>WORLD CUP 2026</span><span>FINAL (SIMULATED)</span></div>
            <div class="trophy">🏆</div>
            <div class="final-title">CHAMPIONSHIP FINAL</div>
            <div class="final-teams">
                <div class="final-team {'champion' if home_team == champion else ''}">
                    <div class="big-badge">{get_flag(home_team)}</div>
                    <div class="final-team-name">{home_team}</div>
                    <div class="final-prob">{home_p:.0f}%</div>
                </div>
                <div class="vs">VS</div>
                <div class="final-team {'champion' if away_team == champion else ''}">
                    <div class="big-badge">{get_flag(away_team)}</div>
                    <div class="final-team-name">{away_team}</div>
                    <div class="final-prob">{away_p:.0f}%</div>
                </div>
            </div>
            <div class="final-probability">
                <div class="final-prob-bar"><div class="final-home-bar" style="width:{home_p}%"></div></div>
                <div class="final-prob-label"><span>{home_p:.0f}%</span><span>{away_p:.0f}%</span></div>
            </div>
            <div class="champion-box">
                <div class="champion-label">PROJECTED CHAMPION</div>
                <div class="champion-name">🏆 {get_flag(champion)} {champion}</div>
            </div>
        </div>
        """

    html = f"""
    <!DOCTYPE html><html><head><meta charset="UTF-8"><style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; padding:15px; background: radial-gradient(circle at center, #0e3324 0%, #071c14 45%, #04120d 100%);
            color:white; font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; overflow-x:auto; }}
    .bracket-wrapper {{ min-width:1500px; padding:30px 30px 45px; border-radius:20px; border:1px solid rgba(212,175,55,0.18);
            background: linear-gradient(180deg, rgba(15,46,33,0.96), rgba(4,18,13,0.98)); box-shadow:0 25px 80px rgba(0,0,0,0.55); }}
    .bracket-top {{ text-align:center; margin-bottom:30px; }}
    .competition {{ color:#d4af37; font-size:10px; font-weight:900; letter-spacing:3px; }}
    .competition-title {{ margin-top:7px; font-size:25px; font-weight:900; letter-spacing:-0.5px;
            background: linear-gradient(90deg, #ffffff, #b7f7d8, #d4af37); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .round-labels {{ display:grid; grid-template-columns:250px 180px 180px 330px 180px 180px 250px; gap:22px; margin-bottom:15px; }}
    .round-label {{ text-align:center; color:#6fae8e; font-size:9px; font-weight:900; letter-spacing:1.5px; text-transform:uppercase; }}
    .final-label {{ color:#d4af37; }}
    .bracket {{ display:grid; grid-template-columns:250px 180px 180px 330px 180px 180px 250px; gap:22px; align-items:stretch; }}
    .round {{ min-height:750px; display:flex; flex-direction:column; justify-content:space-around; position:relative; }}
    .qf {{ padding:65px 0; }}
    .sf {{ padding:190px 0; }}
    .final-column {{ justify-content:center; }}
    .match-card {{ position:relative; width:100%; overflow:hidden; background: linear-gradient(145deg,#123425,#0a1f16);
            border:1px solid #245c40; border-radius:9px; box-shadow:0 8px 20px rgba(0,0,0,0.25);
            transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }}
    .match-card:hover {{ transform: translateY(-3px) scale(1.015); border-color:#d4af37; box-shadow:0 12px 30px rgba(212,175,55,0.18); }}
    .match-header {{ display:flex; justify-content:space-between; padding:7px 10px; background: rgba(255,255,255,0.035);
            color:#7fae95; font-size:7px; font-weight:900; letter-spacing:1px; text-transform:uppercase; }}
    .team-row {{ position:relative; display:flex; align-items:center; justify-content:space-between; min-height:42px;
            padding:5px 10px; border-bottom:1px solid rgba(255,255,255,0.05); }}
    .team-row:last-of-type {{ border-bottom:none; }}
    .team-row.winner {{ background: linear-gradient(90deg, rgba(212,175,55,0.14), transparent); }}
    .team-left {{ display:flex; align-items:center; gap:8px; min-width:0; }}
    .team-badge {{ width:25px; height:25px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
            font-size:14px; }}
    .team-name {{ font-size:10px; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .team-row.winner .team-name {{ color:#f4d58d; }}
    .team-prob {{ color:#7fae95; font-size:8px; font-weight:800; }}
    .winner-mark {{ position:absolute; right:7px; color:#4ade80; font-size:12px; font-weight:900; }}
    .probability-container {{ padding:7px 10px 5px; }}
    .probability-bar {{ height:3px; width:100%; background:#1c4531; border-radius:4px; overflow:hidden; }}
    .probability-home {{ height:100%; background: linear-gradient(90deg,#1fa876,#d4af37); }}
    .probability-labels {{ display:flex; justify-content:space-between; margin-top:3px; color:#5f8b74; font-size:7px; font-weight:700; }}
    .match-footer {{ display:flex; justify-content:space-between; padding:6px 10px 8px; color:#5f8b74; font-size:7px; }}
    .predicted {{ color:#4ade80; font-weight:900; max-width:75px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }}
    .final-card {{ position:relative; padding:25px 20px; border-radius:16px; text-align:center;
            background: radial-gradient(circle at center, rgba(212,175,55,0.14), transparent 65%), linear-gradient(145deg,#123c2a,#081c13);
            border:1px solid rgba(212,175,55,0.4); box-shadow:0 0 45px rgba(212,175,55,0.1); }}
    .final-header {{ display:flex; justify-content:space-between; color:#c9b06a; font-size:7px; font-weight:900; letter-spacing:1.5px; }}
    .trophy {{ font-size:40px; margin-top:10px; }}
    .final-title {{ margin-top:3px; color:#f4d58d; font-size:13px; font-weight:900; letter-spacing:2px; }}
    .final-teams {{ display:flex; align-items:center; justify-content:center; gap:10px; margin-top:25px; }}
    .final-team {{ flex:1; padding:12px 5px; border-radius:10px; }}
    .final-team.champion {{ background: rgba(212,175,55,0.12); box-shadow: inset 0 0 20px rgba(212,175,55,0.06); }}
    .big-badge {{ width:45px; height:45px; margin:auto; display:flex; align-items:center; justify-content:center; font-size:26px; }}
    .final-team-name {{ margin-top:9px; font-size:10px; font-weight:800; }}
    .final-prob {{ margin-top:4px; color:#9fe6c4; font-size:9px; font-weight:800; }}
    .vs {{ color:#6fae8e; font-size:8px; font-weight:900; }}
    .final-probability {{ margin-top:20px; }}
    .final-prob-bar {{ height:5px; background:#1c4531; border-radius:5px; overflow:hidden; }}
    .final-home-bar {{ height:100%; background: linear-gradient(90deg,#1fa876,#f4d58d); }}
    .final-prob-label {{ display:flex; justify-content:space-between; margin-top:5px; color:#7fae95; font-size:8px; }}
    .champion-box {{ margin-top:20px; padding:13px; border-radius:9px; background: rgba(212,175,55,0.08); border:1px solid rgba(212,175,55,0.18); }}
    .champion-label {{ color:#a9925f; font-size:7px; font-weight:900; letter-spacing:1.5px; }}
    .champion-name {{ margin-top:5px; color:#f4d58d; font-size:14px; font-weight:900; }}
    </style></head>
    <body>
        <div class="bracket-wrapper">
            <div class="bracket-top">
                <div class="competition">FIFA WORLD CUP 2026</div>
                <div class="competition-title">KNOCKOUT STAGE (SIMULATED)</div>
            </div>
            <div class="round-labels">
                <div class="round-label">Round of 16</div><div class="round-label">Quarter-Finals</div>
                <div class="round-label">Semi-Finals</div><div class="round-label final-label">Grand Final</div>
                <div class="round-label">Semi-Finals</div><div class="round-label">Quarter-Finals</div>
                <div class="round-label">Round of 16</div>
            </div>
            <div class="bracket">
                <div class="round left-r16">{left_r16_html}</div>
                <div class="round qf left-qf">{left_qf_html}</div>
                <div class="round sf">{left_sf_html}</div>
                <div class="round final-column">{final_html}</div>
                <div class="round sf">{right_sf_html}</div>
                <div class="round qf right-qf">{right_qf_html}</div>
                <div class="round right-r16">{right_r16_html}</div>
            </div>
        </div>
    </body></html>
    """
    components.html(html, height=900, scrolling=True)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(["Match Predictor", "Group Draw & Bracket", "Player Style Clusters", "Beat the AI"])

# ------------------------------------------------------------
# TAB 1 — MATCH PREDICTOR
# ------------------------------------------------------------

with tabs[0]:
    st.header("Match Outcome Predictor")
    st.caption("Elo · FIFA Rank (illustrative) · Head-to-Head · Squad Value · Expected Goals")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams_list, index=teams_list.index("Brazil"), format_func=team_label)
    with col2:
        away_team = st.selectbox("Away Team", teams_list, index=teams_list.index("France"), format_func=team_label)
    neutral_venue = st.checkbox("Neutral Venue Match", value=True)

    if st.button("Predict Match", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams.")
        else:
            result = predictor.predict_match(home_team, away_team, neutral_venue=neutral_venue)
            st.success("Prediction generated.")

            st.markdown("### Team Intelligence")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"#### {get_flag(result['home_team'])} {result['home_team']}")
                st.metric("FIFA Rank", f"#{result['home_rank']}")
                st.metric("Elo Rating", result["home_elo"])
                st.metric("Squad Value", f"€{result['home_market_val']}M")
            with m2:
                st.markdown(f"#### {get_flag(result['away_team'])} {result['away_team']}")
                st.metric("FIFA Rank", f"#{result['away_rank']}")
                st.metric("Elo Rating", result["away_elo"])
                st.metric("Squad Value", f"€{result['away_market_val']}M")

            st.markdown("---")
            st.subheader("Head-to-Head & Expected Goals")
            h2h = result.get("h2h_dominance", 0)
            st.metric(f"H2H Advantage ({result['home_team']} vs {result['away_team']})",
                      f"+{h2h}" if h2h > 0 else str(h2h))
            xg1, xg2 = st.columns(2)
            with xg1:
                st.metric(f"{result['home_team']} xG", result["home_xg"])
            with xg2:
                st.metric(f"{result['away_team']} xG", result["away_xg"])

            st.markdown("---")
            st.subheader("Match Outcome Probabilities")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{result['home_team']} Win", f"{result['home_win_prob']}%")
            c2.metric("Draw", f"{result['draw_prob']}%")
            c3.metric(f"{result['away_team']} Win", f"{result['away_win_prob']}%")

# ------------------------------------------------------------
# TAB 2 — GROUP DRAW + FULL TOURNAMENT (48 teams)
# ------------------------------------------------------------

with tabs[1]:
    st.header("Group Stage Draw")
    st.caption("12 groups of 4, seeded by illustrative rank (pot 1 = strongest in each group) — just like the real 48-team format.")

    group_cols = st.columns(4)
    for i, (gname, gteams) in enumerate(groups.items()):
        with group_cols[i % 4]:
            rows = "".join(f"<div>{get_flag(t)} {t}</div>" for t in gteams)
            st.markdown(f'<div class="group-card"><h5>Group {gname}</h5>{rows}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.header("Full Tournament Simulation")
    st.caption("Simulates every group match, advances the top 2 + best 8 third-place teams to a Round-of-32 knockout, then all the way to a projected champion.")

    if st.button("Simulate Full World Cup", type="primary", use_container_width=True):
        with st.spinner("Playing every group match, then the knockout rounds..."):
            st.session_state["full_results"] = predictor.simulate_full_tournament(groups)

    if "full_results" in st.session_state:
        results = st.session_state["full_results"]

        st.success(f"🏆 Projected Champion: {get_flag(results['Champion'])} {results['Champion']}")

        with st.expander("Group stage standings", expanded=False):
            gcols = st.columns(3)
            for i, (gname, data) in enumerate(results["group_results"].items()):
                with gcols[i % 3]:
                    st.markdown(f"**Group {gname}**")
                    table_rows = [
                        {"Team": f"{get_flag(r['team'])} {r['team']}", "Pld": r["Pld"], "W": r["W"],
                         "D": r["D"], "L": r["L"], "GD": r["GD"], "Pts": r["Pts"]}
                        for r in data["standings"]
                    ]
                    st.dataframe(table_rows, use_container_width=True, hide_index=True)

        with st.expander("🎯 Round of 32 results", expanded=False):
            r32 = results.get("Round of 32", [])
            r32_rows = [
                {"Match": f"{get_flag(m['prediction']['home_team'])} {m['prediction']['home_team']} vs "
                          f"{get_flag(m['prediction']['away_team'])} {m['prediction']['away_team']}",
                 "Winner": f"{get_flag(m['winner'])} {m['winner']}"}
                for m in r32
            ]
            st.dataframe(r32_rows, use_container_width=True, hide_index=True)

        st.markdown("### Knockout bracket: Round of 16 → Final")
        render_champions_league_bracket(results)
    else:
        st.info("Click **Simulate Full World Cup** to draw groups, play the group stage, and run the knockout bracket.")

# ------------------------------------------------------------
# TAB 3 — PLAYER CLUSTERING (interactive: click a point -> radar chart)
# ------------------------------------------------------------

with tabs[2]:
    st.header("Player Style Clustering")
    st.caption("Illustrative sample player profiles, grouped into auto-labeled playing styles via K-Means. Click a point to see that player's full attribute radar.")

    clustered_df, cluster_labels = get_clustered_players()
    clusterer_for_plot = PlayerStyleClusterer()

    axis_col1, axis_col2 = st.columns(2)
    with axis_col1:
        x_axis = st.selectbox("X-axis attribute", ["pace", "shooting", "passing", "dribbling", "defending", "physical"], index=3)
    with axis_col2:
        y_axis = st.selectbox("Y-axis attribute", ["pace", "shooting", "passing", "dribbling", "defending", "physical"], index=4)

    fig = clusterer_for_plot.plot_clusters(clustered_df, x_axis, y_axis)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="cluster_scatter")

    selected_name = None
    points = event.get("selection", {}).get("points", []) if event else []
    if points:
        selected_name = points[0].get("customdata", [None])[0]

    if selected_name:
        st.markdown(f"#### {selected_name} — style profile")
        row = clustered_df[clustered_df["player_name"] == selected_name].iloc[0]
        radar_col, stat_col = st.columns([2, 1])
        with radar_col:
            st.plotly_chart(clusterer_for_plot.plot_player_radar(row), use_container_width=True)
        with stat_col:
            st.markdown(f"**Team:** {get_flag(row['team'])} {row['team']}")
            st.markdown(f"**Style:** {row['style']}")
            for f in ["pace", "shooting", "passing", "dribbling", "defending", "physical"]:
                st.metric(f.capitalize(), int(row[f]))
    else:
        st.info("Click any player point above to open their style radar.")

    st.markdown("---")
    st.markdown("#### Styles found")
    for cid, label in cluster_labels.items():
        members = clustered_df[clustered_df["cluster"] == cid]["player_name"].tolist()
        st.markdown(f"**{label}** — {', '.join(members)}")

# ------------------------------------------------------------
# TAB 4 — BEAT THE AI (gamified prediction challenge)
# ------------------------------------------------------------

with tabs[3]:
    st.header("Beat the AI")
    st.caption("Pick your own Round-of-16 winners from the top 8 seeds, then see how many the model agrees with.")

    default_r16 = [
        ("Argentina", "Ghana"), ("France", "Paraguay"), ("Brazil", "New Zealand"), ("England", "Jamaica"),
        ("Belgium", "Panama"), ("Portugal", "Austria"), ("Netherlands", "Turkey"), ("Spain", "Sweden"),
    ]

    if "user_picks" not in st.session_state:
        st.session_state["user_picks"] = {}

    st.markdown("Pick a winner for each fixture below:")
    for i, (home, away) in enumerate(default_r16):
        pick = st.radio(f"{team_label(home)} vs {team_label(away)}", [home, away], key=f"pick_{i}",
                         horizontal=True, format_func=team_label)
        st.session_state["user_picks"][i] = pick

    if st.button("Compare with the AI", type="primary", use_container_width=True):
        ai_bracket = predictor.simulate_bracket([t for pair in default_r16 for t in pair])
        ai_r16 = ai_bracket.get("Round of 16", [])

        matches, correct = 0, 0
        rows = []
        for i, (home, away) in enumerate(default_r16):
            user_pick = st.session_state["user_picks"].get(i)
            ai_pick = ai_r16[i]["winner"] if i < len(ai_r16) else None
            agree = user_pick == ai_pick
            matches += 1
            correct += int(agree)
            rows.append({"Fixture": f"{home} vs {away}", "Your Pick": user_pick, "AI Pick": ai_pick, "Agree?": "✅" if agree else "❌"})

        st.markdown(f"### You agreed with the AI on {correct}/{matches} picks")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if correct == matches:
            st.balloons()
