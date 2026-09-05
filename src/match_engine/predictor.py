"""
World Cup match predictor.

Design notes (read this before you extend it):
- No network calls. Earlier versions tried to live-fetch FIFA rankings and
  historical results from GitHub CSVs that were unreliable, and silently
  fell back to pure random data when they failed -- which meant the model
  was often trained on noise while the UI presented it as a real prediction.
- Instead, this version generates a synthetic-but-structured match history:
  matchups are simulated using each team's illustrative strength (Elo +
  market value), so the model actually has signal to learn from. This is
  clearly a demo dataset, not real match history -- swap `TEAMS_DB` and
  `generate_synthetic_history()` for a real results CSV to make this "real".
- The model is trained once per instance. Wrap construction in
  st.cache_resource in the Streamlit app so it isn't retrained on every
  click.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
from teams_data import TEAMS_DB, get_rank, get_market_value, get_base_elo, get_flag, team_list, top_n_teams


class WorldCupMatchPredictor:
    def __init__(self, n_synthetic_matches: int = 2500, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
        self.le = LabelEncoder()
        self.teams = team_list()

        self.historical_df = self.generate_synthetic_history(n_synthetic_matches)
        self.elos = self.compute_elo_ratings(self.historical_df)
        self.test_accuracy = None
        self.is_trained = False
        self.train_model()

    # ---------- data ----------

    def generate_synthetic_history(self, n_matches: int) -> pd.DataFrame:
        """Simulate plausible match outcomes from illustrative team strength.

        Goals are drawn from a Poisson distribution whose mean is nudged by
        the Elo gap and market-value gap between the two sides, so stronger
        teams win more often without the result being deterministic.
        """
        rows = []
        for _ in range(n_matches):
            home, away = self.rng.choice(self.teams, size=2, replace=False)
            h_elo, a_elo = get_base_elo(home), get_base_elo(away)
            h_val, a_val = get_market_value(home), get_market_value(away)
            neutral = bool(self.rng.choice([True, False], p=[0.65, 0.35]))

            elo_gap = (h_elo - a_elo) / 400.0
            val_gap = (h_val - a_val) / 1500.0
            home_advantage = 0.0 if neutral else 0.15

            home_lambda = max(0.3, 1.3 + elo_gap + val_gap + home_advantage)
            away_lambda = max(0.3, 1.3 - elo_gap - val_gap)

            home_score = int(self.rng.poisson(home_lambda))
            away_score = int(self.rng.poisson(away_lambda))

            rows.append({
                "home_team": home, "away_team": away,
                "home_score": home_score, "away_score": away_score,
                "neutral": neutral,
            })
        return pd.DataFrame(rows)

    def compute_elo_ratings(self, df: pd.DataFrame) -> dict:
        elos = {t: get_base_elo(t) for t in self.teams}
        for _, row in df.iterrows():
            h, a = row["home_team"], row["away_team"]
            h_score, a_score = row["home_score"], row["away_score"]
            h_exp = 1 / (1 + 10 ** ((elos[a] - elos[h]) / 400))
            a_exp = 1 - h_exp
            if h_score > a_score:
                h_act, a_act = 1.0, 0.0
            elif h_score < a_score:
                h_act, a_act = 0.0, 1.0
            else:
                h_act, a_act = 0.5, 0.5
            k = 20
            elos[h] += k * (h_act - h_exp)
            elos[a] += k * (a_act - a_exp)
        return elos

    # ---------- feature helpers ----------

    def get_team_rank(self, team: str) -> int:
        return get_rank(team)

    def get_team_market_value(self, team: str) -> float:
        return get_market_value(team)

    def get_team_elo(self, team: str) -> float:
        return self.elos.get(team, get_base_elo(team))

    def get_head_to_head_factor(self, home_team: str, away_team: str) -> float:
        matches = self.historical_df[
            ((self.historical_df["home_team"] == home_team) & (self.historical_df["away_team"] == away_team)) |
            ((self.historical_df["home_team"] == away_team) & (self.historical_df["away_team"] == home_team))
        ]
        if len(matches) == 0:
            return 0.0
        home_wins = away_wins = 0
        for _, row in matches.iterrows():
            h_is_home = row["home_team"] == home_team
            if row["home_score"] == row["away_score"]:
                continue
            home_won = (row["home_score"] > row["away_score"]) == h_is_home
            if home_won:
                home_wins += 1
            else:
                away_wins += 1
        return (home_wins - away_wins) / max(1, len(matches))

    def _feature_row(self, home_team: str, away_team: str, neutral: bool):
        return [
            self.get_team_rank(home_team), self.get_team_rank(away_team),
            self.get_team_market_value(home_team), self.get_team_market_value(away_team),
            self.get_team_elo(home_team), self.get_team_elo(away_team),
            self.get_head_to_head_factor(home_team, away_team),
            int(neutral),
        ]

    # ---------- training ----------

    def prepare_data(self, df: pd.DataFrame):
        df = df.copy()
        df["outcome"] = 1  # draw by default
        df.loc[df["home_score"] > df["away_score"], "outcome"] = 0   # home win
        df.loc[df["home_score"] < df["away_score"], "outcome"] = 2   # away win

        feats = [self._feature_row(h, a, n) for h, a, n in
                 zip(df["home_team"], df["away_team"], df["neutral"])]
        X = pd.DataFrame(feats, columns=[
            "home_rank", "away_rank", "home_val", "away_val",
            "home_elo", "away_elo", "h2h", "neutral",
        ])
        y = self.le.fit_transform(df["outcome"])
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data(self.historical_df)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.test_accuracy = round(accuracy_score(y_test, preds) * 100, 1)
        self.is_trained = True

    # ---------- inference ----------

    def calculate_expected_goals(self, h_elo, a_elo, h_val, a_val):
        base = 1.35
        elo_diff = (h_elo - a_elo) / 400.0
        val_diff = (h_val - a_val) / 1500.0
        home_xg = max(0.4, base + elo_diff + val_diff)
        away_xg = max(0.4, base - elo_diff - val_diff)
        return round(home_xg, 1), round(away_xg, 1)

    def predict_match(self, home_team: str, away_team: str, neutral_venue: bool = True):
        h_rank, a_rank = self.get_team_rank(home_team), self.get_team_rank(away_team)
        h_val, a_val = self.get_team_market_value(home_team), self.get_team_market_value(away_team)
        h_elo, a_elo = self.get_team_elo(home_team), self.get_team_elo(away_team)
        h2h = self.get_head_to_head_factor(home_team, away_team)

        feature_cols = ["home_rank", "away_rank", "home_val", "away_val",
                        "home_elo", "away_elo", "h2h", "neutral"]
        features = pd.DataFrame([self._feature_row(home_team, away_team, neutral_venue)], columns=feature_cols)
        probs = self.model.predict_proba(features)[0]
        prob_dict = dict(zip(self.model.classes_, probs))

        home_xg, away_xg = self.calculate_expected_goals(h_elo, a_elo, h_val, a_val)

        return {
            "home_team": home_team, "home_rank": h_rank, "home_market_val": h_val,
            "home_elo": round(h_elo, 1),
            "away_team": away_team, "away_rank": a_rank, "away_market_val": a_val,
            "away_elo": round(a_elo, 1),
            "h2h_dominance": round(h2h, 2),
            "home_xg": home_xg, "away_xg": away_xg,
            "home_win_prob": round(prob_dict.get(0, 0.0) * 100, 2),
            "draw_prob": round(prob_dict.get(1, 0.0) * 100, 2),
            "away_win_prob": round(prob_dict.get(2, 0.0) * 100, 2),
        }

    def simulate_knockout_match(self, team1: str, team2: str):
        prediction = self.predict_match(team1, team2, neutral_venue=True)
        h_prob, a_prob = prediction["home_win_prob"], prediction["away_win_prob"]
        if h_prob == a_prob:
            winner = team1 if self.get_team_elo(team1) >= self.get_team_elo(team2) else team2
        else:
            winner = team1 if h_prob > a_prob else team2
        return {"match": f"{team1} vs {team2}", "prediction": prediction, "winner": winner}

    def simulate_bracket(self, teams: list) -> dict:
        """Model-driven single-elimination knockout simulation. Purely a
        projection from the current model -- NOT real tournament results.
        Round names are chosen from how many teams remain, so this works
        for a Round-of-32 start just as well as a Round-of-16 start."""
        ROUND_NAME_BY_SIZE = {
            32: "Round of 32", 16: "Round of 16", 8: "Quarter-Finals",
            4: "Semi-Finals", 2: "Final",
        }
        rounds = {}
        current = list(teams)
        while len(current) > 1:
            name = ROUND_NAME_BY_SIZE.get(len(current), f"Round of {len(current)}")
            matches = []
            next_round = []
            for i in range(0, len(current), 2):
                if i + 1 >= len(current):
                    next_round.append(current[i])
                    continue
                result = self.simulate_knockout_match(current[i], current[i + 1])
                matches.append(result)
                next_round.append(result["winner"])
            rounds[name] = matches
            current = next_round
        # The renderer expects the Final round as a single match dict, not a list.
        if "Final" in rounds and len(rounds["Final"]) == 1:
            rounds["Final"] = rounds["Final"][0]
        rounds["Champion"] = current[0] if current else None
        return rounds

    # ---------- group stage (48-team format) ----------

    def simulate_scoreline(self, team1: str, team2: str):
        """Simulate an actual scoreline (not just a win-prob) for group-stage
        standings, using the same strength-weighted Poisson approach as the
        synthetic training data."""
        h_elo, a_elo = self.get_team_elo(team1), self.get_team_elo(team2)
        h_val, a_val = self.get_team_market_value(team1), self.get_team_market_value(team2)
        elo_gap = (h_elo - a_elo) / 400.0
        val_gap = (h_val - a_val) / 1500.0
        lam1 = max(0.3, 1.3 + elo_gap + val_gap)
        lam2 = max(0.3, 1.3 - elo_gap - val_gap)
        return int(self.rng.poisson(lam1)), int(self.rng.poisson(lam2))

    def simulate_group_stage(self, groups: dict) -> dict:
        """groups: {"A": [team1, team2, team3, team4], ...}. Round-robin
        every group, return standings + raw match results per group."""
        results = {}
        for group_name, teams in groups.items():
            table = {t: {"team": t, "Pld": 0, "W": 0, "D": 0, "L": 0,
                         "GF": 0, "GA": 0, "GD": 0, "Pts": 0} for t in teams}
            matches = []
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    s1, s2 = self.simulate_scoreline(t1, t2)
                    matches.append({"match": f"{t1} vs {t2}", "score": f"{s1}-{s2}"})
                    for team, gf, ga in ((t1, s1, s2), (t2, s2, s1)):
                        row = table[team]
                        row["Pld"] += 1
                        row["GF"] += gf
                        row["GA"] += ga
                        row["GD"] = row["GF"] - row["GA"]
                        if gf > ga:
                            row["W"] += 1
                            row["Pts"] += 3
                        elif gf < ga:
                            row["L"] += 1
                        else:
                            row["D"] += 1
                            row["Pts"] += 1
            standings = sorted(table.values(), key=lambda r: (-r["Pts"], -r["GD"], -r["GF"]))
            results[group_name] = {"standings": standings, "matches": matches}
        return results

    def build_round_of_32(self, group_results: dict) -> list:
        """Top 2 from each of the 12 groups + best 8 third-placed teams = 32.
        Seeding below is a simplified demo bracket (strongest vs weakest by
        overall standing) -- NOT FIFA's official cross-group qualifier rules."""
        winners, runners_up, thirds = [], [], []
        for group_name, data in group_results.items():
            standings = data["standings"]
            winners.append(standings[0]["team"])
            runners_up.append(standings[1]["team"])
            thirds.append(standings[2])

        thirds_sorted = sorted(thirds, key=lambda r: (-r["Pts"], -r["GD"], -r["GF"]))
        best_thirds = [r["team"] for r in thirds_sorted[:8]]

        qualifiers = winners + runners_up + best_thirds  # 12 + 12 + 8 = 32
        qualifiers.sort(key=lambda t: self.get_team_rank(t))  # strongest first (illustrative)

        n = len(qualifiers)
        bracket = []
        for i in range(n // 2):
            bracket.append(qualifiers[i])
            bracket.append(qualifiers[n - 1 - i])
        return bracket

    def simulate_full_tournament(self, groups: dict) -> dict:
        """Full 48-team pipeline: group stage -> Round of 32 -> ... -> Final."""
        group_results = self.simulate_group_stage(groups)
        round_of_32_teams = self.build_round_of_32(group_results)
        knockout = self.simulate_bracket(round_of_32_teams)
        return {"group_results": group_results, "round_of_32_seed": round_of_32_teams, **knockout}
