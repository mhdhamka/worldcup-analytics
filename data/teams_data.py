"""
Illustrative team reference data for the World Cup 2026 demo project.

IMPORTANT: 48 teams to mirror the real 2026 tournament's expanded format,
but the ranks / market values / base Elo / country codes below are
hand-set, illustrative approximations for demo purposes -- this is NOT
a live feed and NOT a claim about who actually qualified for the real
tournament. Swap this out for a real FIFA ranking + qualification feed
to make it "real".
"""

TEAMS_DB = {
    # name:            (fifa_rank, market_value_millions_eur, base_elo, iso_code)
    "Argentina":     (1, 1250.0, 2120.0, "AR"),
    "France":        (2, 1225.5, 2106.5, "FR"),
    "Brazil":        (3, 1201.0, 2093.0, "BR"),
    "England":       (4, 1176.5, 2079.5, "GB"),
    "Belgium":       (5, 1152.0, 2066.0, "BE"),
    "Portugal":      (6, 1127.5, 2052.5, "PT"),
    "Netherlands":   (7, 1103.0, 2039.0, "NL"),
    "Spain":         (8, 1078.5, 2025.5, "ES"),
    "Croatia":       (9, 1054.0, 2012.0, "HR"),
    "Italy":         (10, 1029.5, 1998.5, "IT"),
    "USA":           (11, 1005.0, 1985.0, "US"),
    "Germany":       (12, 980.5, 1971.5, "DE"),
    "Morocco":       (13, 956.0, 1958.0, "MA"),
    "Switzerland":   (14, 931.5, 1944.5, "CH"),
    "Colombia":      (15, 907.0, 1931.0, "CO"),
    "Mexico":        (16, 882.5, 1917.5, "MX"),
    "Japan":         (17, 858.0, 1904.0, "JP"),
    "Senegal":       (18, 833.5, 1890.5, "SN"),
    "Iran":          (19, 809.0, 1877.0, "IR"),
    "South Korea":   (20, 784.5, 1863.5, "KR"),
    "Denmark":       (21, 760.0, 1850.0, "DK"),
    "Uruguay":       (22, 735.5, 1836.5, "UY"),
    "Australia":     (23, 711.0, 1823.0, "AU"),
    "Ecuador":       (24, 686.5, 1809.5, "EC"),
    "Canada":        (25, 662.0, 1796.0, "CA"),
    "Poland":        (26, 637.5, 1782.5, "PL"),
    "Serbia":        (27, 613.0, 1769.0, "RS"),
    "Tunisia":       (28, 588.5, 1755.5, "TN"),
    "Ukraine":       (29, 564.0, 1742.0, "UA"),
    "Ivory Coast":   (30, 539.5, 1728.5, "CI"),
    "Nigeria":       (31, 515.0, 1715.0, "NG"),
    "Qatar":         (32, 490.5, 1701.5, "QA"),
    "Saudi Arabia":  (33, 466.0, 1688.0, "SA"),
    "Cameroon":      (34, 441.5, 1674.5, "CM"),
    "Chile":         (35, 417.0, 1661.0, "CL"),
    "Peru":          (36, 392.5, 1647.5, "PE"),
    "Costa Rica":    (37, 368.0, 1634.0, "CR"),
    "Egypt":         (38, 343.5, 1620.5, "EG"),
    "Algeria":       (39, 319.0, 1607.0, "DZ"),
    "Norway":        (40, 294.5, 1593.5, "NO"),
    "Sweden":        (41, 270.0, 1580.0, "SE"),
    "Turkey":        (42, 245.5, 1566.5, "TR"),
    "Austria":       (43, 221.0, 1553.0, "AT"),
    "Panama":        (44, 196.5, 1539.5, "PA"),
    "Jamaica":       (45, 172.0, 1526.0, "JM"),
    "New Zealand":   (46, 147.5, 1512.5, "NZ"),
    "Ghana":         (47, 123.0, 1499.0, "GH"),
    "Paraguay":      (48, 98.5, 1485.5, "PY"),
}

DEFAULT_RANK = 60
DEFAULT_MARKET_VALUE = 80.0
DEFAULT_ELO = 1470.0


def get_rank(team: str) -> int:
    return TEAMS_DB.get(team, (DEFAULT_RANK, None, None, None))[0]


def get_market_value(team: str) -> float:
    return TEAMS_DB.get(team, (None, DEFAULT_MARKET_VALUE, None, None))[1]


def get_base_elo(team: str) -> float:
    return TEAMS_DB.get(team, (None, None, DEFAULT_ELO, None))[2]


def get_flag(team: str) -> str:
    """Return a flag emoji built from the team's ISO 3166-1 alpha-2 code."""
    code = TEAMS_DB.get(team, (None, None, None, None))[3]
    if not code:
        return "🏳️"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code.upper())


def team_list():
    return list(TEAMS_DB.keys())


def top_n_teams(n: int):
    """Teams sorted by illustrative rank, best first."""
    return sorted(TEAMS_DB.keys(), key=get_rank)[:n]


def seeded_groups(n_groups: int = 12, group_size: int = 4):
    """Build groups using a standard pot-based draw: teams are split into
    `group_size` pots by rank (pot 1 = best team in each group, etc.) so
    strong teams are spread across groups instead of clustering into one
    'group of death' -- closer to how a real World Cup draw works."""
    ranked = sorted(TEAMS_DB.keys(), key=get_rank)
    groups = {chr(ord("A") + i): [] for i in range(n_groups)}
    for pot in range(group_size):
        pot_teams = ranked[pot * n_groups:(pot + 1) * n_groups]
        for i, team in enumerate(pot_teams):
            groups[chr(ord("A") + i)].append(team)
    return groups
