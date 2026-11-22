import csv
import math
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

FEATURE_NAMES = [
    'home_elo_pre',
    'away_elo_pre',
    'neutral',
    'home_form_goals_scored_last5',
    'home_form_goals_conceded_last5',
    'away_form_goals_scored_last5',
    'away_form_goals_conceded_last5',
]
LAMBDA_SCALE_HOME = 1.0
LAMBDA_SCALE_AWAY = 1.0
OVERDISPERSION_SIGMA = 0.0


@lru_cache(maxsize=1)
def linear_coefficients():
    history = pd.read_csv(
        ROOT / 'loc' / 'matches_truoc_wc_2026_elo_features.csv',
        encoding='utf-8-sig',
    )
    x_history = history[FEATURE_NAMES].copy()
    x_history['neutral'] = x_history['neutral'].astype(str).str.strip().str.casefold().isin(
        {'true', '1', 'yes'}
    ).astype(float)

    coefficients = {}
    for target, label in (('home_score', 'home'), ('away_score', 'away')):
        model = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=0.01, max_iter=3000),
        )
        model.fit(x_history, history[target].astype(float))
        scaler = model.named_steps['standardscaler']
        regressor = model.named_steps['poissonregressor']
        scale = np.asarray(scaler.scale_, dtype=float)
        coef = np.asarray(regressor.coef_, dtype=float).ravel()
        weights = coef / scale
        bias = float(
            regressor.intercept_
            - np.dot(coef, np.asarray(scaler.mean_, dtype=float) / scale)
        )
        coefficients[label] = (weights, bias)
    return coefficients


W_HOME, B_HOME = linear_coefficients()['home']
W_AWAY, B_AWAY = linear_coefficients()['away']

@lru_cache(maxsize=1)
def load_groups():
    groups = defaultdict(list)
    path = ROOT / 'dataset' / 'mo_phong' / 'group_stages.csv'
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('group'):
                parts = line.split(';')
                if len(parts) >= 3:
                    groups[parts[0]].append(parts[2])
    return dict(sorted(groups.items()))

@lru_cache(maxsize=1)
def load_knockout_fixtures():
    path = ROOT / 'dataset' / 'mo_phong' / 'fixtures_knockout_wc2026.csv'
    fixtures = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixtures.append(row)
    return fixtures

@lru_cache(maxsize=1)
def load_third_place_table():
    path = ROOT / 'dataset' / 'mo_phong' / 'third_place_combinations_wc2026.csv'
    table = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row['qualified_groups']] = row
    return table

@lru_cache(maxsize=1)
def load_confederations():
    path = ROOT / 'dataset' / 'mo_phong' / 'FIFA_confederations.csv'
    table = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            table[row['nation'].strip()] = row['confederation'].strip()
    return table


# --- Bốc thăm vòng bảng theo luật FIFA -------------------------------------
# 1. 48 đội chia 4 nhóm hạt giống (pot), mỗi nhóm 12 đội, xếp theo Elo.
#    Pot 1 gồm 3 chủ nhà cộng 9 đội Elo cao nhất còn lại.
# 2. Mỗi bảng nhận đúng một đội từ mỗi pot.
# 3. Một bảng không được có hai đội cùng liên đoàn, trừ UEFA (tối đa 2).
# 4. Chủ nhà cố định vị trí: Mexico A1, Canada B1, Mỹ D1.
DRAW_GROUPS = list('ABCDEFGHIJKL')
HOST_TEAMS = ('Mexico', 'Canada', 'United States')
HOST_SLOTS = {'A': 'Mexico', 'B': 'Canada', 'D': 'United States'}
CONFEDERATION_LIMITS = {'UEFA': 2}
DEFAULT_CONFEDERATION_LIMIT = 1

_current_draw = {'groups': None, 'seed': None, 'payload': None}


def build_pots():
    """Chia 48 đội thành 4 pot theo Elo, pot 1 ưu tiên chủ nhà."""
    teams = [row['team'] for row in ranked_teams()]
    others = [team for team in teams if team not in HOST_TEAMS]
    return {
        1: list(HOST_TEAMS) + others[:9],
        2: others[9:21],
        3: others[21:33],
        4: others[33:45],
    }


def _greedy_attempt(rng, pots, pot_of, confederations):
    """Một lần xếp tham lam: trả về 12 bảng, hoặc None nếu bí giữa đường.

    Đội bị ràng buộc chặt (UEFA, tối đa 2 suất/bảng) được xếp trước để
    không còn lại nhóm cuối cùng toàn đội châu Âu. Bí thì bỏ cả lần thử,
    vì mỗi lần thử chỉ tốn vài chục micro-giây.
    """
    groups = {g: [] for g in DRAW_GROUPS}
    conf_count = {g: defaultdict(int) for g in DRAW_GROUPS}
    pot_used = {g: set() for g in DRAW_GROUPS}

    def place(group, team):
        groups[group].append(team)
        conf_count[group][confederations.get(team, '—')] += 1
        pot_used[group].add(pot_of[team])

    for g, team in HOST_SLOTS.items():
        place(g, team)

    rest_pot1 = [team for team in pots[1] if team not in HOST_SLOTS.values()]
    rng.shuffle(rest_pot1)
    free_groups = [g for g in DRAW_GROUPS if g not in HOST_SLOTS]
    for g, team in zip(free_groups, rest_pot1):
        place(g, team)

    remaining = [team for pot in (2, 3, 4) for team in pots[pot]]
    rng.shuffle(remaining)
    remaining.sort(key=lambda team: 0 if confederations.get(team) == 'UEFA' else 1)

    for team in remaining:
        conf = confederations.get(team, '—')
        limit = CONFEDERATION_LIMITS.get(conf, DEFAULT_CONFEDERATION_LIMIT)
        pot = pot_of[team]
        options = [
            g for g in DRAW_GROUPS
            if pot not in pot_used[g] and conf_count[g][conf] + 1 <= limit
        ]
        if not options:
            return None
        place(options[int(rng.integers(len(options)))], team)

    return groups


def draw_groups(seed=None, attempts=400):
    """Bốc thăm 12 bảng; trả về payload đầy đủ cho giao diện.

    Một số ít lá thăm rơi vào nhánh tìm kiếm cạn ngân sách, khi đó thử lại
    bằng một nhánh khác sinh từ cùng seed — kết quả vẫn tất định theo seed.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 10 ** 9))
    confederations = load_confederations()
    pots = build_pots()
    pot_of = {team: pot for pot, team_list in pots.items() for team in team_list}

    groups = None
    for attempt in range(attempts):
        rng = np.random.default_rng(seed + attempt * 7919)
        groups = _greedy_attempt(rng, pots, pot_of, confederations)
        if groups is not None:
            break

    if groups is None:
        raise ValueError('Không xếp được bảng thoả luật FIFA với lá thăm này.')

    elo = {row['team']: row['elo'] for row in ranked_teams()}

    return {
        'seed': int(seed),
        'pots': {str(pot): pots[pot] for pot in pots},
        'groups': {
            g: [
                {
                    'team': team,
                    'pot': pot_of[team],
                    'confederation': confederations.get(team, '—'),
                    'elo': elo.get(team),
                    'host': team in HOST_SLOTS.values(),
                }
                for team in sorted(groups[g], key=lambda t: pot_of[t])
            ]
            for g in DRAW_GROUPS
        },
    }


def real_groups_payload():
    """12 bảng thật của World Cup 2026, kèm pot/liên đoàn để hiển thị."""
    confederations = load_confederations()
    pots = build_pots()
    pot_of = {team: pot for pot, team_list in pots.items() for team in team_list}
    elo = {row['team']: row['elo'] for row in ranked_teams()}
    return {
        'seed': None,
        'real': True,
        'pots': {str(pot): pots[pot] for pot in pots},
        'groups': {
            g: [
                {
                    'team': team,
                    'pot': pot_of.get(team),
                    'confederation': confederations.get(team, '—'),
                    'elo': elo.get(team),
                    'host': team in HOST_TEAMS,
                }
                for team in teams
            ]
            for g, teams in load_groups().items()
        },
    }


def check_draw(payload):
    """Kiểm tra lại một lá thăm có đúng luật không (dùng cho test)."""
    confederations = load_confederations()
    problems = []
    seen = []
    for g, rows in payload['groups'].items():
        if len(rows) != 4:
            problems.append(f'Bảng {g} có {len(rows)} đội')
        pots = [row['pot'] for row in rows]
        if sorted(pots) != [1, 2, 3, 4]:
            problems.append(f'Bảng {g} sai pot: {pots}')
        counts = defaultdict(int)
        for row in rows:
            counts[row['confederation']] += 1
            seen.append(row['team'])
        for conf, used in counts.items():
            limit = CONFEDERATION_LIMITS.get(conf, DEFAULT_CONFEDERATION_LIMIT)
            if used > limit:
                problems.append(f'Bảng {g} có {used} đội {conf}')
    if len(seen) != 48 or len(set(seen)) != 48:
        problems.append('Số đội không phải 48 duy nhất')
    for g, team in HOST_SLOTS.items():
        if payload['groups'][g][0]['team'] != team:
            problems.append(f'Chủ nhà {team} không ở {g}1')
    return problems


def set_current_draw(payload):
    _current_draw['groups'] = {
        g: [row['team'] for row in rows] for g, rows in payload['groups'].items()
    }
    _current_draw['seed'] = payload['seed']
    _current_draw['payload'] = payload


def current_draw_groups():
    return _current_draw['groups']


def current_draw_seed():
    return _current_draw['seed']


def current_draw_payload():
    return _current_draw['payload']


def clear_current_draw():
    _current_draw['groups'] = None
    _current_draw['seed'] = None
    _current_draw['payload'] = None


# --- Bể đội để đổi suất dự giải -------------------------------------------
# 48 đội trong giải chỉ có Elo trong wc_2026_da_loc.csv. Muốn đổi một đội
# ngoài giải vào (Ý, Đan Mạch, Nigeria...) thì phải tính lại Elo cho họ trên
# CÙNG thang với 48 đội, tức là chạy lại Elo từ 1930 trên results.csv bằng
# đúng bảng K của wc.ipynb.
TIER_1_WORLD_CUP = {'FIFA World Cup'}
TIER_2_CONTINENTAL = {
    'UEFA Euro', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup',
    'Gold Cup', 'CONCACAF Championship', 'Oceania Nations Cup', 'Confederations Cup',
}
TIER_3_QUALIFIERS = {
    'FIFA World Cup qualification', 'UEFA Euro qualification',
    'African Cup of Nations qualification', 'AFC Asian Cup qualification',
    'Gold Cup qualification', 'CONCACAF Championship qualification',
    'Copa América qualification', 'Oceania Nations Cup qualification',
    'UEFA Nations League', 'CONCACAF Nations League',
    'CONCACAF Nations League qualification',
}
TIER_4_REGIONAL = {
    'CECAFA Cup', 'COSAFA Cup', 'COSAFA Cup qualification', 'WAFF Championship',
    'Amílcar Cabral Cup', 'All-African Games', 'UDEAC Cup', 'UNIFFAC Cup',
    'West African Cup', 'Nile Basin Tournament', 'African Friendship Games',
    'Gulf Cup', 'Arab Cup', 'Arab Cup qualification', 'SAFF Cup',
    'AFF Championship', 'AFF Championship qualification', 'EAFF Championship',
    'EAFF Championship qualification', 'ASEAN Championship',
    'ASEAN Championship qualification', 'AFC Challenge Cup',
    'AFC Challenge Cup qualification', 'Asian Games', 'CAFA Nations Cup',
    'Southeast Asian Games', 'South Asian Games', 'Dynasty Cup',
    'Pacific Games', 'South Pacific Games', 'Melanesia Cup',
    'Indian Ocean Island Games', 'Afro-Asian Games', 'British Home Championship',
    'Nordic Championship', 'Baltic Cup', 'Balkan Cup',
    'Central European International Cup', 'CFU Caribbean Cup',
    'CFU Caribbean Cup qualification', 'UNCAF Cup',
    'Central American and Caribbean Games', 'Pan American Championship',
    'CCCF Championship', 'Bolivarian Games', 'NAFC Championship', 'Olympic Games',
}
TIER_5_FRIENDLY = {'Friendly', 'FIFA Series', 'CONCACAF Series'}

TEAM_NAME_ALIASES = {
    'Czechia': 'Czech Republic',
    'Türkiye': 'Turkey',
    'IR Iran': 'Iran',
    'USA': 'United States',
    'Congo DR': 'DR Congo',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
}


def _k_factor(tournament):
    if tournament in TIER_1_WORLD_CUP:
        return 60
    if tournament in TIER_2_CONTINENTAL:
        return 50
    if tournament in TIER_3_QUALIFIERS:
        return 40
    if tournament in TIER_4_REGIONAL:
        return 30
    if tournament in TIER_5_FRIENDLY:
        return 20
    return 15


def _is_competitive(tournament):
    """Đội tuyển thật là đội từng đá vòng loại hoặc giải châu lục.

    Loại các đội không thuộc FIFA như Basque Country, Jersey, Guernsey —
    họ chỉ đá giao hữu/khu vực nhưng Elo vẫn cao.
    """
    return (
        tournament in TIER_1_WORLD_CUP
        or tournament in TIER_2_CONTINENTAL
        or tournament in TIER_3_QUALIFIERS
    )


@lru_cache(maxsize=1)
def load_pool_states():
    """Elo và phong độ 5 trận cuối của mọi đội tuyển tại 11/06/2026."""
    from collections import deque as _deque

    path = ROOT / 'dataset' / 'du_lieu_tho' / 'results.csv'
    df = pd.read_csv(path, encoding='utf-8')
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= pd.Timestamp('1930-01-01'))
            & (df['date'] < pd.Timestamp('2026-06-11'))]
    df = df[~df['tournament'].str.contains('CONIFA', case=False, na=False)]
    for column in ('home_team', 'away_team'):
        df[column] = df[column].replace(TEAM_NAME_ALIASES)
    df = df.sort_values('date', kind='stable')

    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: _deque(maxlen=5))
    competitive = defaultdict(bool)
    last_match = {}

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hs, as_ = float(row.home_score), float(row.away_score)
        neutral = str(row.neutral).strip().casefold() in {'true', '1', 'yes'}

        home_elo, away_elo = elo[home], elo[away]
        advantage = 0 if neutral else 100
        expected = 1.0 / (1.0 + 10 ** (-((home_elo + advantage) - away_elo) / 400))
        actual = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        margin = abs(hs - as_)
        multiplier = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        k = _k_factor(row.tournament)

        elo[home] = home_elo + k * multiplier * (actual - expected)
        elo[away] = away_elo + k * multiplier * ((1 - actual) - (1 - expected))
        form[home].append((hs, as_))
        form[away].append((as_, hs))

        last_match[home] = row.date
        last_match[away] = row.date
        if _is_competitive(row.tournament):
            competitive[home] = True
            competitive[away] = True

    states = {}
    for team, value in elo.items():
        matches = list(form[team])
        scored = [m[0] for m in matches] or [1.3]
        conceded = [m[1] for m in matches] or [1.3]
        scored = (scored * 5)[:5]
        conceded = (conceded * 5)[:5]
        states[team] = {
            'elo': float(value),
            'gf': [float(x) for x in scored],
            'ga': [float(x) for x in conceded],
            'competitive': bool(competitive[team]),
            'active': last_match.get(team) is not None
            and last_match[team] >= pd.Timestamp('2024-01-01'),
        }
    return states


def swap_pool(limit=20):
    """48 đội trong giải + `limit` đội ngoài giải có Elo cao nhất."""
    states = load_pool_states()
    tournament_teams = {team for teams in load_groups().values() for team in teams}
    ranked = sorted(states.items(), key=lambda item: item[1]['elo'], reverse=True)

    inside = [
        {'team': team, 'elo': round(state['elo'], 1), 'in_tournament': True}
        for team, state in ranked if team in tournament_teams
    ]
    outside = [
        {'team': team, 'elo': round(state['elo'], 1), 'in_tournament': False}
        for team, state in ranked
        if team not in tournament_teams and state['competitive'] and state['active']
    ][:limit]
    return {'inside': inside, 'outside': outside}


@lru_cache(maxsize=1)
def load_initial_states():
    path = ROOT / 'loc' / 'wc_2026_da_loc.csv'
    wc_features = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        wc_features = list(csv.DictReader(f))
    
    states = {}
    groups = load_groups()
    all_teams = sorted({team for tlist in groups.values() for team in tlist})
    
    for team in all_teams:
        matches = [m for m in wc_features if m['home_team'] == team or m['away_team'] == team]
        if not matches:
            raise ValueError(f'Thiếu đặc trưng Elo/phong độ cho đội {team}')
        first = matches[0]
        if first['home_team'] == team:
            elo = float(first['home_elo_pre'])
            gf = float(first['home_form_goals_scored_last5'])
            ga = float(first['home_form_goals_conceded_last5'])
        else:
            elo = float(first['away_elo_pre'])
            gf = float(first['away_form_goals_scored_last5'])
            ga = float(first['away_form_goals_conceded_last5'])
        states[team] = {
            'elo': elo,
            'gf': [gf] * 5,
            'ga': [ga] * 5,
        }
    return states

@lru_cache(maxsize=1)
def ranked_teams():
    """Các đội World Cup 2026, xếp theo Elo trước giải (cao nhất trước)."""
    states = load_initial_states()
    return [
        {'team': team, 'elo': round(state['elo'], 1)}
        for team, state in sorted(
            states.items(), key=lambda item: item[1]['elo'], reverse=True
        )
    ]


def selectable_teams(limit=40):
    """Top `limit` đội có Elo cao nhất trong tập dữ liệu World Cup 2026."""
    return ranked_teams()[:limit]


def selectable_team_names(limit=40):
    return [row['team'] for row in selectable_teams(limit)]


def sample_poisson(lam, rng):
    if lam <= 0:
        return 0
    return int(rng.poisson(lam))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def predict_lambda(home_team, away_team, states, neutral=True):
    home = states[home_team]
    away = states[away_team]
    x = [
        home['elo'], away['elo'], 1.0 if neutral else 0.0,
        sum(home['gf']) / 5.0, sum(home['ga']) / 5.0,
        sum(away['gf']) / 5.0, sum(away['ga']) / 5.0,
    ]
    lh = float(np.clip(
        math.exp(dot(x, W_HOME) + B_HOME) * LAMBDA_SCALE_HOME,
        0.05,
        12.0,
    ))
    la = float(np.clip(
        math.exp(dot(x, W_AWAY) + B_AWAY) * LAMBDA_SCALE_AWAY,
        0.05,
        12.0,
    ))
    return lh, la


def predict_match(home_team, away_team, max_goals=12, restrict_to_top=True):
    states = load_initial_states()
    if home_team not in states:
        raise KeyError(home_team)
    if away_team not in states:
        raise KeyError(away_team)
    if home_team == away_team:
        raise ValueError('Hai đội phải khác nhau')
    if restrict_to_top:
        allowed = set(selectable_team_names())
        if home_team not in allowed or away_team not in allowed:
            raise ValueError('Chỉ được chọn trong 40 đội có Elo cao nhất.')

    home_lambda, away_lambda = predict_lambda(home_team, away_team, states, neutral=False)
    scores = {}
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    home_probs = [math.exp(-home_lambda) * home_lambda ** g / math.factorial(g)
                   for g in range(max_goals + 1)]
    away_probs = [math.exp(-away_lambda) * away_lambda ** g / math.factorial(g)
                   for g in range(max_goals + 1)]
    total = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            probability = home_probs[h] * away_probs[a]
            scores[f'{h}-{a}'] = probability
            total += probability
            if h > a:
                home_win += probability
            elif h == a:
                draw += probability
            else:
                away_win += probability

    if total <= 0:
        total = 1.0
    scores = {key: value / total for key, value in scores.items()}
    home_win /= total
    draw /= total
    away_win /= total

    top_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_lambda': round(home_lambda, 3),
        'away_lambda': round(away_lambda, 3),
        'outcomes': {
            'home_win': round(home_win * 100, 2),
            'draw': round(draw * 100, 2),
            'away_win': round(away_win * 100, 2),
        },
        'top_scores': [
            {'score': score, 'probability': round(probability * 100, 2)}
            for score, probability in top_scores
        ],
    }

def expected_home(home_team, away_team, states):
    diff = states[home_team]['elo'] - states[away_team]['elo']
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def update_state(home_team, away_team, home_goals, away_goals, states, actual_home=None):
    home, away = states[home_team], states[away_team]
    exp_h = expected_home(home_team, away_team, states)
    if actual_home is None:
        if home_goals > away_goals:
            act_h = 1.0
        elif home_goals < away_goals:
            act_h = 0.0
        else:
            act_h = 0.5
    else:
        act_h = actual_home
    
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        mult = 1.0
    elif diff == 2:
        mult = 1.5
    else:
        mult = (11.0 + diff) / 8.0
        
    delta = 60.0 * mult * (act_h - exp_h)
    home['elo'] += delta
    away['elo'] -= delta
    
    home['gf'] = home['gf'][1:] + [float(home_goals)]
    home['ga'] = home['ga'][1:] + [float(away_goals)]
    away['gf'] = away['gf'][1:] + [float(away_goals)]
    away['ga'] = away['ga'][1:] + [float(home_goals)]

def play_match(home_team, away_team, states, match_id, round_name, rng):
    host_teams = {'Mexico', 'United States', 'Canada'}
    neutral = not (home_team in host_teams)
    lh, la = predict_lambda(home_team, away_team, states, neutral=neutral)
    if OVERDISPERSION_SIGMA:
        shock = math.exp(rng.normal(0.0, OVERDISPERSION_SIGMA))
        lh *= shock
        la *= shock
    hg = sample_poisson(lh, rng)
    ag = sample_poisson(la, rng)
    extra_time = False
    penalties = False
    winner = None
    regulation_home_goals = hg
    regulation_away_goals = ag
    penalty_actual_home = None

    if round_name in ('R32', 'R16', 'QF', 'SF', '3rd', 'Final'):
        if hg == ag:
            extra_time = True
            hg += sample_poisson(lh * 0.35, rng)
            ag += sample_poisson(la * 0.35, rng)
        if hg == ag:
            penalties = True
            elo_probability = expected_home(home_team, away_team, states)
            penalty_probability = min(0.52, max(0.48, elo_probability))
            winner = home_team if rng.random() < penalty_probability else away_team
            penalty_actual_home = 1.0 if winner == home_team else 0.0
        else:
            winner = home_team if hg > ag else away_team
    else:
        if hg > ag:
            winner = home_team
        elif ag > hg:
            winner = away_team
        else:
            winner = None

    if penalties:
        update_state(
            home_team,
            away_team,
            regulation_home_goals,
            regulation_away_goals,
            states,
            actual_home=penalty_actual_home,
        )
    else:
        update_state(home_team, away_team, hg, ag, states)
    return {
        'match_id': match_id,
        'round': round_name,
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': hg,
        'away_goals': ag,
        'extra_time': extra_time,
        'penalties': penalties,
        'winner': winner,
    }

def _rank_group(teams, stats, records):
    ranked = []
    point_values = sorted({stats[team]['points'] for team in teams}, reverse=True)

    for points in point_values:
        tied = [team for team in teams if stats[team]['points'] == points]
        if len(tied) == 1:
            ranked.extend(tied)
            continue

        tied_set = set(tied)
        h2h = {}
        for team in tied:
            h2h_points = gf = ga = 0
            for match in records:
                if match.get('round') != 'Vòng bảng':
                    continue
                home = match['home_team']
                away = match['away_team']
                if home not in tied_set or away not in tied_set:
                    continue
                hg = match['home_goals']
                ag = match['away_goals']
                if home == team:
                    gf += hg
                    ga += ag
                    h2h_points += 3 if hg > ag else (1 if hg == ag else 0)
                elif away == team:
                    gf += ag
                    ga += hg
                    h2h_points += 3 if ag > hg else (1 if ag == hg else 0)
            h2h[team] = (h2h_points, gf - ga, gf)

        tied.sort(key=lambda team: (
            h2h[team],
            stats[team]['gf'] - stats[team]['ga'],
            stats[team]['gf'],
        ), reverse=True)
        ranked.extend(tied)
    return ranked

def _normalize_swaps(swap_in, swap_out):
    """Chấp nhận một cặp hoặc danh sách nhiều cặp đổi suất."""

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(',') if part.strip()]
        return [str(part).strip() for part in value if str(part).strip()]

    ins, outs = as_list(swap_in), as_list(swap_out)
    if len(ins) != len(outs):
        raise ValueError('Số đội đưa vào và số đội bị thay phải bằng nhau.')
    if len(set(ins)) != len(ins):
        raise ValueError('Một đội không thể được đưa vào hai lần.')
    if len(set(outs)) != len(outs):
        raise ValueError('Một đội không thể bị thay hai lần.')
    return ins, outs


def _apply_team_swap(groups, swap_ins=None, swap_outs=None):
    """Đổi suất dự giải.

    `swap_out` phải đang ở trong giải, `swap_in` phải đang ở NGOÀI giải —
    đội đã có suất thì không thể được đưa vào lần nữa.
    """
    swap_ins = swap_ins or []
    swap_outs = swap_outs or []
    if not swap_ins and not swap_outs:
        return groups, None

    groups = {group: list(teams) for group, teams in groups.items()}
    pairs = []

    for swap_in, swap_out in zip(swap_ins, swap_outs):
        if swap_in == swap_out:
            raise ValueError('Đội đưa vào và đội bị thay phải khác nhau.')

        group_out = next((g for g, teams in groups.items() if swap_out in teams), None)
        if group_out is None:
            raise ValueError(f'Đội {swap_out} không có trong giải đấu.')
        index_out = groups[group_out].index(swap_out)

        group_in = next((g for g, teams in groups.items() if swap_in in teams), None)
        if group_in is not None:
            raise ValueError(
                f'Đội {swap_in} đã có suất trong giải — chỉ đổi được với đội ngoài giải.'
            )
        groups[group_out][index_out] = swap_in

        pairs.append({'in': swap_in, 'out': swap_out})

    return groups, {'pairs': pairs}


def simulate_full_tournament(seed=None, swap_in=None, swap_out=None):
    swap_ins, swap_outs = _normalize_swaps(swap_in, swap_out)
    rng = np.random.default_rng(seed)
    base_states = load_initial_states()
    states = {
        t: {
            'elo': s['elo'],
            'gf': list(s['gf']),
            'ga': list(s['ga']),
        }
        for t, s in base_states.items()
    }

    # Đội ngoài giải được đổi vào cần Elo/phong độ lấy từ bể đội.
    pool = load_pool_states() if swap_ins else {}
    for team in swap_ins:
        if team in states:
            continue
        source = pool.get(team)
        if source is None:
            raise ValueError(f'Đội {team} không có trong bể đội.')
        states[team] = {
            'elo': source['elo'],
            'gf': list(source['gf']),
            'ga': list(source['ga']),
        }
    
    groups, applied_swap = _apply_team_swap(
        current_draw_groups() or load_groups(), swap_ins, swap_outs
    )
    team_to_group = {t: g for g, teams in groups.items() for t in teams}
    knockout_fixtures = load_knockout_fixtures()
    third_place_table = load_third_place_table()

    group_matches = []
    standings = {}

    for g_id, teams in groups.items():
        stats = {t: {'points': 0, 'gf': 0, 'ga': 0} for t in teams}
        pairs = list(combinations(teams, 2))
        for idx, (t1, t2) in enumerate(pairs, 1):
            m = play_match(t1, t2, states, f'{g_id}-M{idx}', 'Vòng bảng', rng)
            group_matches.append(m)
            hg, ag = m['home_goals'], m['away_goals']
            stats[t1]['gf'] += hg
            stats[t1]['ga'] += ag
            stats[t2]['gf'] += ag
            stats[t2]['ga'] += hg
            if hg > ag:
                stats[t1]['points'] += 3
            elif ag > hg:
                stats[t2]['points'] += 3
            else:
                stats[t1]['points'] += 1
                stats[t2]['points'] += 1
        
        ranked = _rank_group(teams, stats, group_matches)
        standings[g_id] = {'ranked': ranked, 'stats': stats}

    third_places = []
    for g_id, val in standings.items():
        team = val['ranked'][2]
        s = val['stats'][team]
        third_places.append((team, s['points'], s['gf'] - s['ga'], s['gf'], g_id))
    
    third_places.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    top8_third = third_places[:8]
    
    q_groups = ''.join(sorted(team_to_group[t[0]] for t in top8_third))
    third_opt = third_place_table.get(q_groups)
    slot_to_team = {}
    if third_opt:
        for first_slot in ('1A', '1B', '1D', '1E', '1G', '1I', '1K', '1L'):
            tgt_g = third_opt.get(first_slot)
            cand = next(t[0] for t in top8_third if team_to_group[t[0]] == tgt_g)
            slot_to_team[f'T{first_slot}'] = cand
    else:
        for idx, t in enumerate(top8_third):
            slot_to_team[f'3rd_{idx}'] = t[0]

    winners = {}
    losers = {}
    knockout_matches = []

    def resolve(slot):
        if slot in slot_to_team:
            return slot_to_team[slot]
        if slot.startswith('T'):
            return slot_to_team.get(slot, top8_third[0][0])
        if slot.startswith('W'):
            return winners[slot]
        if slot.startswith('RU'):
            return losers[slot]
        if len(slot) == 2 and slot[0].isdigit():
            r = int(slot[0]) - 1
            g = slot[1]
            return standings[g]['ranked'][r]
        if slot.startswith('3'):
            return slot_to_team.get(f'T{slot[1:]}', top8_third[0][0])
        return list(states.keys())[0]

    for fix in knockout_fixtures:
        r_name = fix['round']
        m_id = fix['match_id']
        home = resolve(fix['home_slot'])
        away = resolve(fix['away_slot'])
        res = play_match(home, away, states, m_id, r_name, rng)
        knockout_matches.append(res)
        w = res['winner']
        l = away if w == home else home
        winners[f'W{m_id[1:]}'] = w
        losers[f'RU{m_id[1:]}'] = l

    champion = next(m['winner'] for m in knockout_matches if m['round'] == 'Final')
    return {
        'group_matches': group_matches,
        'standings': standings,
        'top8_third': [t[0] for t in top8_third],
        'knockout_matches': knockout_matches,
        'champion': champion,
        'swap': applied_swap,
        'draw_seed': current_draw_seed(),
    }

def run_monte_carlo(runs=100, swap_in=None, swap_out=None):
    stage_counts = defaultdict(lambda: defaultdict(int))
    for i in range(runs):
        res = simulate_full_tournament(seed=None, swap_in=swap_in, swap_out=swap_out)
        for g_id, data in res['standings'].items():
            for t in data['ranked'][:2]:
                stage_counts[t]['R32'] += 1
        for t in res['top8_third']:
            stage_counts[t]['R32'] += 1

        for m in res['knockout_matches']:
            r = m['round']
            w = m['winner']
            if r == 'R32':
                stage_counts[w]['R16'] += 1
            elif r == 'R16':
                stage_counts[w]['QF'] += 1
            elif r == 'QF':
                stage_counts[w]['SF'] += 1
            elif r == 'SF':
                stage_counts[w]['Final'] += 1
            elif r == 'Final':
                stage_counts[w]['Champion'] += 1

    summary = []
    for team, counts in stage_counts.items():
        summary.append({
            'team': team,
            'R32': round(counts['R32'] / runs * 100, 1),
            'R16': round(counts['R16'] / runs * 100, 1),
            'QF': round(counts['QF'] / runs * 100, 1),
            'SF': round(counts['SF'] / runs * 100, 1),
            'Final': round(counts['Final'] / runs * 100, 1),
            'Champion': round(counts['Champion'] / runs * 100, 1),
        })
    summary.sort(key=lambda x: (x['Champion'], x['Final'], x['SF'], x['QF']), reverse=True)
    return summary
