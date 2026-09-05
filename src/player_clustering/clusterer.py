"""
Player playing-style clustering.

Notes:
- SAMPLE_PLAYERS below is an illustrative demo dataset (approximate,
  hand-set 0-100 style ratings loosely inspired by each player's real
  profile) -- it is NOT sourced from any official stats provider. Swap
  `SAMPLE_PLAYERS` for a real dataset (e.g. FBref/Opta export) to make
  this reflect real season stats.
- Clusters are auto-labeled from their centroid so the chart reads like
  "Playmakers" / "Ball-Winners" instead of an opaque cluster number.
"""
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["pace", "shooting", "passing", "dribbling", "defending", "physical"]

SAMPLE_PLAYERS = pd.DataFrame([
    {"player_name": "Kylian Mbappe",      "team": "France",      "pace": 97, "shooting": 90, "passing": 80, "dribbling": 92, "defending": 36, "physical": 78},
    {"player_name": "Erling Haaland",     "team": "Norway",      "pace": 89, "shooting": 93, "passing": 66, "dribbling": 80, "defending": 45, "physical": 88},
    {"player_name": "Jude Bellingham",    "team": "England",     "pace": 80, "shooting": 82, "passing": 84, "dribbling": 86, "defending": 68, "physical": 79},
    {"player_name": "Vinicius Jr.",       "team": "Brazil",      "pace": 95, "shooting": 82, "passing": 78, "dribbling": 93, "defending": 29, "physical": 68},
    {"player_name": "Rodri",              "team": "Spain",       "pace": 60, "shooting": 68, "passing": 90, "dribbling": 78, "defending": 86, "physical": 82},
    {"player_name": "Toni Kroos",         "team": "Germany",     "pace": 52, "shooting": 70, "passing": 93, "dribbling": 78, "defending": 60, "physical": 68},
    {"player_name": "Virgil van Dijk",    "team": "Netherlands", "pace": 74, "shooting": 55, "passing": 71, "dribbling": 62, "defending": 92, "physical": 90},
    {"player_name": "Ruben Dias",         "team": "Portugal",    "pace": 68, "shooting": 42, "passing": 68, "dribbling": 58, "defending": 90, "physical": 86},
    {"player_name": "Bukayo Saka",        "team": "England",     "pace": 88, "shooting": 80, "passing": 82, "dribbling": 87, "defending": 45, "physical": 68},
    {"player_name": "Lamine Yamal",       "team": "Spain",       "pace": 91, "shooting": 78, "passing": 84, "dribbling": 90, "defending": 32, "physical": 60},
    {"player_name": "Casemiro",           "team": "Brazil",      "pace": 58, "shooting": 62, "passing": 76, "dribbling": 68, "defending": 88, "physical": 86},
    {"player_name": "Declan Rice",        "team": "England",     "pace": 68, "shooting": 60, "passing": 80, "dribbling": 72, "defending": 85, "physical": 84},
    {"player_name": "Pedri",              "team": "Spain",       "pace": 72, "shooting": 68, "passing": 90, "dribbling": 88, "defending": 55, "physical": 60},
    {"player_name": "Jamal Musiala",      "team": "Germany",     "pace": 84, "shooting": 76, "passing": 82, "dribbling": 91, "defending": 40, "physical": 64},
    {"player_name": "Achraf Hakimi",      "team": "Morocco",     "pace": 93, "shooting": 68, "passing": 76, "dribbling": 80, "defending": 68, "physical": 76},
    {"player_name": "Theo Hernandez",     "team": "France",      "pace": 90, "shooting": 66, "passing": 72, "dribbling": 78, "defending": 70, "physical": 82},
    {"player_name": "Christian Pulisic",  "team": "USA",         "pace": 84, "shooting": 74, "passing": 78, "dribbling": 84, "defending": 40, "physical": 62},
    {"player_name": "Alphonso Davies",    "team": "Canada",      "pace": 96, "shooting": 62, "passing": 70, "dribbling": 82, "defending": 66, "physical": 74},
    {"player_name": "Kevin De Bruyne",    "team": "Belgium",     "pace": 74, "shooting": 84, "passing": 94, "dribbling": 86, "defending": 58, "physical": 76},
    {"player_name": "Alexis Mac Allister","team": "Argentina",   "pace": 72, "shooting": 74, "passing": 86, "dribbling": 80, "defending": 66, "physical": 70},
])


class PlayerStyleClusterer:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = {}

    def cluster_players(self, df: pd.DataFrame, feature_columns: list = FEATURES) -> pd.DataFrame:
        df = df.copy()
        scaled = self.scaler.fit_transform(df[feature_columns])
        df["cluster"] = self.kmeans.fit_predict(scaled)
        self.cluster_labels = self._label_clusters(df, feature_columns)
        df["style"] = df["cluster"].map(self.cluster_labels)
        return df

    def _label_clusters(self, df: pd.DataFrame, feature_columns: list) -> dict:
        """Turn each cluster's centroid into a human-readable style label.

        Rather than just taking each cluster's single highest raw stat (pace
        tends to win that contest for almost every attacker), this looks at
        which feature each cluster is MOST elevated on relative to the
        overall player pool -- that's what actually distinguishes one style
        group from another.
        """
        labels = {}
        means = df.groupby("cluster")[feature_columns].mean()
        overall_mean = df[feature_columns].mean()
        deviation = means - overall_mean

        for cluster_id, row in means.iterrows():
            dev = deviation.loc[cluster_id]
            standout = dev.idxmax()

            if row["defending"] > 78:
                labels[cluster_id] = "Ball-Winning Defender"
            elif standout == "passing" or (row["passing"] > 82 and row["defending"] < 70):
                labels[cluster_id] = "Deep-Lying Playmaker"
            elif standout in ("dribbling", "pace") and row["defending"] < 50:
                labels[cluster_id] = "Pacey Dribbler / Winger"
            elif standout == "pace" and row["physical"] > 75:
                labels[cluster_id] = "Attacking Full-Back"
            elif standout == "shooting":
                labels[cluster_id] = "Poacher / Finisher"
            elif standout == "physical":
                labels[cluster_id] = "Box-to-Box Enforcer"
            else:
                labels[cluster_id] = f"Style Group {cluster_id}"
        return labels

    def plot_clusters(self, df: pd.DataFrame, x_col: str = "dribbling", y_col: str = "defending"):
        """Scatter plot styled for the app's dark theme. Designed to be used
        with st.plotly_chart(..., on_select="rerun", selection_mode="points")
        so clicking / lasso-selecting a point is captured by the caller and
        can drive a detail view (e.g. a radar chart of that player)."""
        fig = px.scatter(
            df, x=x_col, y=y_col, color="style",
            hover_name="player_name", hover_data=["team"],
            custom_data=["player_name"],
            title=None,
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#f4d58d", "#4ade80", "#f87171", "#a78bfa", "#fb923c"],
        )
        fig.update_traces(
            marker=dict(size=16, line=dict(width=1.5, color="rgba(255,255,255,0.6)"), opacity=0.9),
        )
        fig.update_layout(
            legend_title_text="Style",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e5e7eb"),
            margin=dict(t=10, l=10, r=10, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            clickmode="event+select",
        )
        return fig

    def plot_player_radar(self, player_row: pd.Series, feature_columns: list = FEATURES):
        """A radar/spider chart of one player's attribute profile -- used as
        the detail view when a point is selected on the cluster scatter."""
        import plotly.graph_objects as go
        values = [player_row[f] for f in feature_columns] + [player_row[feature_columns[0]]]
        labels = [f.capitalize() for f in feature_columns] + [feature_columns[0].capitalize()]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels, fill="toself",
            line=dict(color="#f4d58d", width=2),
            fillcolor="rgba(244,213,141,0.25)",
            name=player_row.get("player_name", "Player"),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.15)", tickfont=dict(color="#9fb3d1")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.15)", tickfont=dict(color="#e5e7eb")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e5e7eb"),
            showlegend=False,
            margin=dict(t=30, l=30, r=30, b=30),
        )
        return fig
