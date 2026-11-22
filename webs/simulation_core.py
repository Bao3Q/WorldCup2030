import csv
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hệ số Poisson đã huấn luyện tối ưu
W_HOME = [0.0016269, -0.0017083, -0.1118924, 0.0619732, -0.0237561, 0.0705482, 0.1239089]
B_HOME = 0.1412473
W_AWAY = [-0.001756, 0.0017745, 0.2090362, 0.0977128, 0.1459086, 0.0458015, 0.0067001]
B_AWAY = -0.4833897

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

def load_knockout_fixtures():
    path = ROOT / 'dataset' / 'mo_phong' / 'fixtures_knockout_wc2026.csv'
    fixtures = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixtures.append(row)
    return fixtures

def load_third_place_table():
    path = ROOT / 'dataset' / 'mo_phong' / 'third_place_combinations_wc2026.csv'
    table = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row['qualified_groups']] = row
    return table

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
            elo = 1800.0
            gf, ga = 1.2, 1.2
        else:
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

def sample_poisson(lam, rng):
    if lam <= 0:
        return 0
    if lam > 30:
        val = rng.gauss(lam, math.sqrt(lam))
        return max(0, int(round(val)))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def predict_lambda(home_team, away_team, states):
    home = states[home_team]
    away = states[away_team]
    x = [
        home['elo'], away['elo'], 1.0,
        sum(home['gf']) / 5.0, sum(home['ga']) / 5.0,
        sum(away['gf']) / 5.0, sum(away['ga']) / 5.0,
    ]
    lh = min(max(math.exp(dot(x, W_HOME) + B_HOME), 0.05), 12.0)
    la = min(max(math.exp(dot(x, W_AWAY) + B_AWAY), 0.05), 12.0)
    return lh, la

def expected_home(home_team, away_team, states):
    diff = states[home_team]['elo'] - states[away_team]['elo']
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def update_state(home_team, away_team, home_goals, away_goals, states):
    home, away = states[home_team], states[away_team]
    exp_h = expected_home(home_team, away_team, states)
    if home_goals > away_goals:
        act_h = 1.0
    elif home_goals < away_goals:
        act_h = 0.0
    else:
        act_h = 0.5
    
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
    lh, la = predict_lambda(home_team, away_team, states)
    hg = sample_poisson(lh, rng)
    ag = sample_poisson(la, rng)
    extra_time = False
    penalties = False
    winner = None

    if round_name in ('R32', 'R16', 'QF', 'SF', '3rd', 'Final'):
        if hg == ag:
            extra_time = True
            hg += sample_poisson(lh * 0.35, rng)
            ag += sample_poisson(la * 0.35, rng)
        if hg == ag:
            penalties = True
            winner = home_team if rng.random() < expected_home(home_team, away_team, states) else away_team
        else:
            winner = home_team if hg > ag else away_team
    else:
        if hg > ag:
            winner = home_team
        elif ag > hg:
            winner = away_team
        else:
            winner = None

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
    # stats: dict[team] -> {'points': p, 'gf': gf, 'ga': ga}
    overall_key = lambda team: (
        stats[team]['points'],
        stats[team]['gf'] - stats[team]['ga'],
        stats[team]['gf'],
    )
    ranked = sorted(teams, key=overall_key, reverse=True)

    # Nếu vẫn bằng sau điểm, hiệu số và bàn thắng, xét thành tích đối đầu (H2H)
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and overall_key(ranked[j]) == overall_key(ranked[i]):
            j += 1
        tied = ranked[i:j]
        if len(tied) > 1:
            tied_set = set(tied)
            h2h = {}
            for team in tied:
                p = gf = ga = 0
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
                        p += 3 if hg > ag else (1 if hg == ag else 0)
                    elif away == team:
                        gf += ag
                        ga += hg
                        p += 3 if ag > hg else (1 if ag == hg else 0)
                h2h[team] = (p, gf - ga, gf)
            tied.sort(key=lambda t: h2h[t], reverse=True)
            ranked[i:j] = tied
        i = j
    return ranked

def simulate_full_tournament(seed=None):
    rng = random.Random(seed)
    base_states = load_initial_states()
    states = {
        t: {
            'elo': s['elo'],
            'gf': list(s['gf']),
            'ga': list(s['ga']),
        }
        for t, s in base_states.items()
    }
    
    groups = load_groups()
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
    }

def run_monte_carlo(runs=100):
    stage_counts = defaultdict(lambda: defaultdict(int))
    for i in range(runs):
        res = simulate_full_tournament(seed=None)
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
