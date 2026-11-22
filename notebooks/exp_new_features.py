"""Thử nghiệm: thêm feature mới có cải thiện mô hình không?

Khung đánh giá giống train.ipynb: walk-forward trên 12 giải, mỗi giải chỉ
huấn luyện bằng dữ liệu TRƯỚC ngày khai mạc. Chỉ số chính là RPS (thấp hơn
tốt hơn) cho 1X2, kèm MAE bàn và độ chính xác tỉ số cho mô hình Poisson.

Mọi feature mới đều tính point-in-time trên trục thời gian từ 1872:
trạng thái của một trận chỉ dùng kết quả các trận TRƯỚC ngày đó.
"""

from __future__ import annotations

import bisect
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'dataset' / 'du_lieu_tho' / 'results.csv'

TRAIN_START = pd.Timestamp('2018-06-14')
WC_START = pd.Timestamp('2026-06-11')

TEAM_NAME_ALIASES = {
    'Czechia': 'Czech Republic',
    'Türkiye': 'Turkey',
    'IR Iran': 'Iran',
    'USA': 'United States',
    'Congo DR': 'DR Congo',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
}

TOURNAMENTS = {
    'Copa America 2019': ('2019-06-14', '2019-07-07', 'Copa América'),
    'Asian Cup 2019': ('2019-01-05', '2019-02-01', 'AFC Asian Cup'),
    'AFCON 2019': ('2019-06-21', '2019-07-19', 'African Cup of Nations'),
    'Euro 2021': ('2021-06-11', '2021-07-11', 'UEFA Euro'),
    'Copa America 2021': ('2021-06-13', '2021-07-10', 'Copa América'),
    'AFCON 2021': ('2022-01-09', '2022-02-06', 'African Cup of Nations'),
    'WC 2022': ('2022-11-20', '2022-12-18', 'FIFA World Cup'),
    'Asian Cup 2023': ('2024-01-12', '2024-02-10', 'AFC Asian Cup'),
    'AFCON 2023': ('2024-01-13', '2024-02-11', 'African Cup of Nations'),
    'Copa America 2024': ('2024-06-20', '2024-07-14', 'Copa América'),
    'Euro 2024': ('2024-06-14', '2024-07-14', 'UEFA Euro'),
    'WC 2026': ('2026-06-11', '2026-07-19', 'FIFA World Cup'),
}

# --- Elo: bản sao của wc.ipynb để Elo khớp từng đơn vị với loc/*.csv ---
TIER_1 = {'FIFA World Cup'}
TIER_2 = {'UEFA Euro', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup',
          'Gold Cup', 'CONCACAF Championship', 'Oceania Nations Cup', 'Confederations Cup'}
TIER_3 = {'FIFA World Cup qualification', 'UEFA Euro qualification',
          'African Cup of Nations qualification', 'AFC Asian Cup qualification',
          'Gold Cup qualification', 'CONCACAF Championship qualification',
          'Copa América qualification', 'Oceania Nations Cup qualification',
          'UEFA Nations League', 'CONCACAF Nations League',
          'CONCACAF Nations League qualification'}
TIER_4 = {'CECAFA Cup', 'COSAFA Cup', 'COSAFA Cup qualification', 'WAFF Championship',
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
          'CCCF Championship', 'Bolivarian Games', 'NAFC Championship', 'Olympic Games'}
TIER_5 = {'Friendly', 'FIFA Series', 'CONCACAF Series'}
HOME_ADVANTAGE = 100


def tournament_tier(name: str) -> int:
    """0 = giao hữu ... 4 = World Cup (dùng làm feature ngữ cảnh)."""
    if name in TIER_1:
        return 4
    if name in TIER_2:
        return 3
    if name in TIER_3:
        return 2
    if name in TIER_5:
        return 0
    if name in TIER_4:
        return 1
    return -1


def k_factor(name: str) -> float:
    """Đúng bảng K của wc.ipynb, kể cả nhánh mặc định K=15."""
    return {4: 60, 3: 50, 2: 40, 1: 30, 0: 20}.get(tournament_tier(name), 15)


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    for col in ('home_team', 'away_team'):
        df[col] = df[col].replace(TEAM_NAME_ALIASES)
    df['date'] = pd.to_datetime(df['date'])
    df = df[~df['tournament'].str.contains('CONIFA', case=False, na=False)].copy()
    df['neutral'] = df['neutral'].astype(str).str.strip().str.casefold().isin({'true', '1', 'yes'})
    df = df.sort_values('date', kind='stable').reset_index(drop=True)
    df['tier'] = df['tournament'].map(tournament_tier)
    return df


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Elo tuần tự giống hệt wc.ipynb (cập nhật ngay trong cùng ngày)."""
    elo = defaultdict(lambda: 1500.0)
    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))
    elo_start = pd.Timestamp('1930-01-01')
    for i, row in enumerate(df.itertuples(index=False)):
        h, a = row.home_team, row.away_team
        h_elo, a_elo = elo[h], elo[a]
        if row.date < elo_start:          # wc.ipynb chỉ chạy Elo từ 1930
            home_pre[i], away_pre[i] = h_elo, a_elo
            continue
        home_pre[i], away_pre[i] = h_elo, a_elo
        adv = 0 if row.neutral else HOME_ADVANTAGE
        exp_home = 1 / (1 + 10 ** (-((h_elo + adv) - a_elo) / 400))
        hs, as_ = float(row.home_score), float(row.away_score)
        act_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        gd = abs(hs - as_)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        k = k_factor(row.tournament)
        elo[h] = h_elo + k * mult * (act_home - exp_home)
        elo[a] = a_elo + k * mult * ((1 - act_home) - (1 - exp_home))
    out = df.copy()
    out['home_elo_pre'] = home_pre
    out['away_elo_pre'] = away_pre
    return out


def baseline_lambda_model(df: pd.DataFrame):
    """Fit mapping Elo -> λ bàn trên dữ liệu TRƯỚC 2018, dùng cho feature
    'phong độ vượt kỳ vọng'. Không chạm vào tập test."""
    early = df[df['date'] < TRAIN_START]
    X = np.column_stack([
        early['home_elo_pre'], early['away_elo_pre'], early['neutral'].astype(float)
    ])
    scale = StandardScaler().fit(X)
    Xs = scale.transform(X)
    home_model = PoissonRegressor(alpha=0.01, max_iter=2000).fit(Xs, early['home_score'])
    away_model = PoissonRegressor(alpha=0.01, max_iter=2000).fit(Xs, early['away_score'])
    return scale, home_model, away_model


def expected_goals(scale, home_model, away_model, home_elo, away_elo, neutral):
    row = scale.transform([[home_elo, away_elo, neutral]])
    return (float(home_model.predict(row)[0]), float(away_model.predict(row)[0]))


def build_features(df: pd.DataFrame, scale, home_model, away_model) -> pd.DataFrame:
    """Một lượt qua toàn bộ lịch sử, tính mọi feature trước mỗi trận."""
    elo_dates = defaultdict(list)
    elo_values = defaultdict(list)

    last_played = {}                       # team -> date
    played_dates = defaultdict(list)        # team -> list[date]
    career_matches = defaultdict(int)      # team -> tổng số trận đã đá
    recent = defaultdict(lambda: deque(maxlen=10))   # team -> (gf, ga, pts, adj_s, adj_c, opp_elo, is_comp)
    recent_comp = defaultdict(lambda: deque(maxlen=10))  # chỉ trận chính thức
    h2h = defaultdict(lambda: deque(maxlen=8))       # (a, b) -> (date, gd_for_a)

    rows = []
    for day, group in df.groupby('date', sort=True):
        # --- PASS 1: lấy Elo trước trận (đã tính sẵn), đồng thời cập nhật Elo ---
        day_elo = {}
        for idx, row in group.iterrows():
            home, away = row['home_team'], row['away_team']
            h_elo, a_elo = float(row['home_elo_pre']), float(row['away_elo_pre'])
            day_elo[idx] = (h_elo, a_elo)

            adv = 0 if row['neutral'] else HOME_ADVANTAGE
            exp_home = 1 / (1 + 10 ** (-((h_elo + adv) - a_elo) / 400))
            hs, as_ = float(row['home_score']), float(row['away_score'])
            act_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            gd = abs(hs - as_)
            mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
            k = k_factor(row['tournament'])
            elo_after = {
                home: h_elo + k * mult * (act_home - exp_home),
                away: a_elo + k * mult * ((1 - act_home) - (1 - exp_home)),
            }
            day_elo[idx] = (h_elo, a_elo, elo_after[home], elo_after[away])

        # --- PASS 2: feature đọc trạng thái TRƯỚC ngày, rồi mới cập nhật ---
        for idx, row in group.iterrows():
            home, away = row['home_team'], row['away_team']
            h_elo, a_elo = day_elo[idx][:2]

            def form_block(team, opp_elo):
                hist = list(recent[team])
                if not hist:
                    return dict(s5=1.3, c5=1.3, s10=1.3, c10=1.3,
                                sw=1.3, cw=1.3, ppg=1.3, adj_s=0.0, adj_c=0.0,
                                std_s=1.3, std_c=1.3, opp_elo=1500.0, comp_ppg=1.3)
                n5 = hist[-5:]
                n10 = hist
                gf10_all = np.array([m[0] for m in n10], float)
                ga10_all = np.array([m[1] for m in n10], float)
                opp_elo5 = np.array([m[5] for m in n10[-5:]], float)
                comp = [m for m in recent_comp[team]]
                comp_ppg = float(np.mean([m[2] for m in comp])) if comp else 1.3
                w = np.array([0.75 ** (len(n5) - 1 - i) for i in range(len(n5))])
                w10 = np.array([0.85 ** (len(n10) - 1 - i) for i in range(len(n10))])
                gf5 = np.array([m[0] for m in n5], float)
                ga5 = np.array([m[1] for m in n5], float)
                gf10 = np.array([m[0] for m in n10], float)
                ga10 = np.array([m[1] for m in n10], float)
                pts = np.array([m[2] for m in n10], float)
                adj_s = np.array([m[3] for m in n10], float)
                adj_c = np.array([m[4] for m in n10], float)
                return dict(
                    std_s=float(gf10_all.std()), std_c=float(ga10_all.std()),
                    opp_elo=float(opp_elo5.mean()), comp_ppg=comp_ppg,
                    s5=float(gf5.mean()), c5=float(ga5.mean()),
                    s10=float(gf10.mean()), c10=float(ga10.mean()),
                    sw=float((gf5 * w).sum() / w.sum()), cw=float((ga5 * w).sum() / w.sum()),
                    ppg=float((pts[-5:] * w).sum() / w.sum()),
                    adj_s=float((adj_s * w10).sum() / w10.sum()),
                    adj_c=float((adj_c * w10).sum() / w10.sum()),
                )

            hf = form_block(home, a_elo)
            af = form_block(away, h_elo)

            # ngày nghỉ và mật độ thi đấu
            def rest_days(team):
                if team not in last_played:
                    return 60.0
                return float(min((day - last_played[team]).days, 60))

            def congestion(team):
                dates = played_dates[team]
                if not dates:
                    return 0.0
                cut = day - pd.Timedelta(days=30)
                return float(len(dates) - bisect.bisect_left(dates, cut))

            # đối đầu trực tiếp
            meetings = list(h2h[(home, away)])
            h2h_n = float(len(meetings))
            h2h_gd = float(np.mean([m[1] for m in meetings])) if meetings else 0.0
            h2h_ppg = float(np.mean([1.0 if m[1] > 0 else (0.5 if m[1] == 0 else 0.0)
                                     for m in meetings])) if meetings else 0.5

            # động lượng Elo: thay đổi trong 365 ngày, đỉnh 3 năm
            def elo_dyn(team, now_elo):
                dates = elo_dates[team]
                if not dates:
                    return 0.0, 0.0
                cut = day - pd.Timedelta(days=365)
                i = bisect.bisect_left(dates, cut)
                base = elo_values[team][i - 1] if i > 0 else 1500.0
                cut3 = day - pd.Timedelta(days=3 * 365)
                j = bisect.bisect_left(dates, cut3)
                window = elo_values[team][j:]
                peak = (max(window) - now_elo) if window else 0.0
                return float(now_elo - base), float(peak)

            h_d365, h_peak = elo_dyn(home, h_elo)
            a_d365, a_peak = elo_dyn(away, a_elo)

            rows.append({
                'date': day, 'home_team': home, 'away_team': away,
                'tournament': row['tournament'],
                'home_score': row['home_score'], 'away_score': row['away_score'],
                # baseline
                'home_elo_pre': h_elo, 'away_elo_pre': a_elo,
                'neutral': float(row['neutral']),
                'home_form_goals_scored_last5': hf['s5'],
                'home_form_goals_conceded_last5': hf['c5'],
                'away_form_goals_scored_last5': af['s5'],
                'away_form_goals_conceded_last5': af['c5'],
                # rest / congestion
                'home_rest_days': rest_days(home), 'away_rest_days': rest_days(away),
                'home_matches_30d': congestion(home), 'away_matches_30d': congestion(away),
                # h2h
                'h2h_n': h2h_n, 'h2h_gd': h2h_gd, 'h2h_ppg': h2h_ppg,
                # form mở rộng
                'home_form_scored_10': hf['s10'], 'home_form_conceded_10': hf['c10'],
                'away_form_scored_10': af['s10'], 'away_form_conceded_10': af['c10'],
                'home_form_scored_w': hf['sw'], 'home_form_conceded_w': hf['cw'],
                'away_form_scored_w': af['sw'], 'away_form_conceded_w': af['cw'],
                'home_ppg5': hf['ppg'], 'away_ppg5': af['ppg'],
                # elo dynamics / structure
                'home_elo_d365': h_d365, 'away_elo_d365': a_d365,
                'home_elo_peak_gap': h_peak, 'away_elo_peak_gap': a_peak,
                'elo_diff': h_elo - a_elo, 'elo_sum': h_elo + a_elo,
                'tier': float(row['tier']),
                # độ ổn định + sức mạnh đối thủ + phong độ chính thức
                'home_goal_std': hf['std_s'], 'away_goal_std': af['std_s'],
                'home_conceded_std': hf['std_c'], 'away_conceded_std': af['std_c'],
                'home_opp_elo5': hf['opp_elo'], 'away_opp_elo5': af['opp_elo'],
                'home_comp_ppg': hf['comp_ppg'], 'away_comp_ppg': af['comp_ppg'],
                # kinh nghiệm: số trận đã đá (proxy cho số lần khoác áo)
                'home_career_matches': float(career_matches[home]),
                'away_career_matches': float(career_matches[away]),
                'home_matches_2y': float(len(played_dates[home]) - bisect.bisect_left(
                    played_dates[home], day - pd.Timedelta(days=730))),
                'away_matches_2y': float(len(played_dates[away]) - bisect.bisect_left(
                    played_dates[away], day - pd.Timedelta(days=730))),
                # phong độ vượt kỳ vọng theo Elo
                'home_adj_scored': hf['adj_s'], 'home_adj_conceded': hf['adj_c'],
                'away_adj_scored': af['adj_s'], 'away_adj_conceded': af['adj_c'],
            })

        # --- PASS 3: cập nhật trạng thái sau khi cả ngày đã đá xong ---
        for idx, row in group.iterrows():
            home, away = row['home_team'], row['away_team']
            hs, as_ = float(row['home_score']), float(row['away_score'])
            h_elo, a_elo, h_after, a_after = day_elo[idx]
            exp_h, exp_a = expected_goals(scale, home_model, away_model, h_elo, a_elo,
                                          float(row['neutral']))
            h_pts = 3.0 if hs > as_ else (1.0 if hs == as_ else 0.0)
            a_pts = 3.0 if as_ > hs else (1.0 if hs == as_ else 0.0)
            is_comp = row['tournament'] not in TIER_5
            recent[home].append((hs, as_, h_pts, hs - exp_h, as_ - exp_a, a_elo, is_comp))
            recent[away].append((as_, hs, a_pts, as_ - exp_a, hs - exp_h, h_elo, is_comp))
            if is_comp:
                recent_comp[home].append((hs, as_, h_pts))
                recent_comp[away].append((as_, hs, a_pts))
            last_played[home] = day
            last_played[away] = day
            played_dates[home].append(day)
            played_dates[away].append(day)
            career_matches[home] += 1
            career_matches[away] += 1
            h2h[(home, away)].append((day, hs - as_))
            h2h[(away, home)].append((day, as_ - hs))
            elo_dates[home].append(day)
            elo_values[home].append(h_after)
            elo_dates[away].append(day)
            elo_values[away].append(a_after)

    return pd.DataFrame(rows)


# ----------------------------- đánh giá -------------------------------------
BASE = ['home_elo_pre', 'away_elo_pre', 'neutral',
        'home_form_goals_scored_last5', 'home_form_goals_conceded_last5',
        'away_form_goals_scored_last5', 'away_form_goals_conceded_last5']

GROUPS = {
    'rest+congestion': ['home_rest_days', 'away_rest_days',
                        'home_matches_30d', 'away_matches_30d'],
    'h2h': ['h2h_n', 'h2h_gd', 'h2h_ppg'],
    'form_last10': ['home_form_scored_10', 'home_form_conceded_10',
                    'away_form_scored_10', 'away_form_conceded_10'],
    'form_weighted': ['home_form_scored_w', 'home_form_conceded_w',
                      'away_form_scored_w', 'away_form_conceded_w'],
    'points_per_game': ['home_ppg5', 'away_ppg5'],
    'elo_momentum': ['home_elo_d365', 'away_elo_d365',
                     'home_elo_peak_gap', 'away_elo_peak_gap'],
    'elo_structure': ['elo_diff', 'elo_sum'],
    'tournament_tier': ['tier'],
    'form_vs_expectation': ['home_adj_scored', 'home_adj_conceded',
                            'away_adj_scored', 'away_adj_conceded'],
    'experience': ['home_career_matches', 'away_career_matches',
                   'home_matches_2y', 'away_matches_2y'],
    'consistency': ['home_goal_std', 'away_goal_std',
                    'home_conceded_std', 'away_conceded_std'],
    'schedule_strength': ['home_opp_elo5', 'away_opp_elo5'],
    'competitive_form': ['home_comp_ppg', 'away_comp_ppg'],
}


def rps(y, proba):
    """Ranked probability score cho 3 cửa (nhà, hoà, khách)."""
    obs = np.zeros((len(y), 3))
    obs[np.arange(len(y)), y] = 1.0
    cum_p = np.cumsum(proba, axis=1)[:, :2]
    cum_o = np.cumsum(obs, axis=1)[:, :2]
    return 0.5 * ((cum_p - cum_o) ** 2).sum(axis=1)


def log_loss3(y, proba, eps=1e-12):
    return -np.log(np.clip(proba[np.arange(len(y)), y], eps, 1.0))


def outcome_index(hs, as_):
    hs = np.asarray(hs, float)
    as_ = np.asarray(as_, float)
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


def score_matrix(lam_h, lam_a, max_goals=12):
    from scipy.stats import poisson
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    return m / m.sum()


def outcome_probs(matrix):
    home = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()
    return np.array([home, draw, away])


def exact_hit(hs, as_, lam_h, lam_a):
    out = []
    for h, a, lh, la in zip(hs, as_, lam_h, lam_a):
        m = score_matrix(lh, la)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        out.append(1.0 if (i == h and j == a) else 0.0)
    return np.array(out)


def run_experiment(features: pd.DataFrame, cols, min_rows=15, C=1.0):
    """Trả về dict chỉ số tổng hợp + vector RPS từng trận để bootstrap."""
    per_match = []
    for name, (start, end, tournament) in TOURNAMENTS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        test = features[(features['date'] >= start) & (features['date'] <= end)
                        & (features['tournament'] == tournament)]
        train = features[features['date'] < start]
        if len(test) < min_rows or len(train) < 200:
            continue

        Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
        ytr = outcome_index(train['home_score'], train['away_score'])
        yte = outcome_index(test['home_score'], test['away_score'])

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=C))
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)

        lam_h = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.01, max_iter=3000))
        lam_a = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.01, max_iter=3000))
        lam_h.fit(Xtr, train['home_score'].to_numpy(float))
        lam_a.fit(Xtr, train['away_score'].to_numpy(float))
        ph = np.clip(lam_h.predict(Xte), 0.05, 12)
        pa = np.clip(lam_a.predict(Xte), 0.05, 12)

        # 1X2 suy ra từ ma trận Poisson (cách notebook dùng khi mô phỏng)
        pois_proba = np.array([outcome_probs(score_matrix(lh, la)) for lh, la in zip(ph, pa)])

        per_match.append(pd.DataFrame({
            'tournament': name,
            'rps_poisson': rps(yte, pois_proba),
            'acc_poisson': (pois_proba.argmax(axis=1) == yte).astype(float),
            'rps': rps(yte, proba),
            'logloss': log_loss3(yte, proba),
            'correct': (proba.argmax(axis=1) == yte).astype(float),
            'exact': exact_hit(test['home_score'].to_numpy(), test['away_score'].to_numpy(), ph, pa),
            'mae': (np.abs(ph - test['home_score'].to_numpy())
                    + np.abs(pa - test['away_score'].to_numpy())) / 2,
        }))
    return pd.concat(per_match, ignore_index=True)


def summarize(per_match: pd.DataFrame) -> dict:
    return {
        'RPS': per_match['rps'].mean(),
        'RPS_pois': per_match['rps_poisson'].mean(),
        'LogLoss': per_match['logloss'].mean(),
        '1X2': per_match['correct'].mean(),
        '1X2_pois': per_match['acc_poisson'].mean(),
        'Exact': per_match['exact'].mean(),
        'MAE': per_match['mae'].mean(),
    }


def bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, n=10000, seed=7):
    """P(b tốt hơn a) bằng bootstrap ghép cặp trên RPS từng trận."""
    d = a['rps'].to_numpy() - b['rps'].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(means.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), float((means > 0).mean())


def main():
    df = add_elo(load_matches())
    scale, hm, am = baseline_lambda_model(df)
    print(f'Dữ liệu: {len(df):,} trận, {df["date"].min().date()} → {df["date"].max().date()}')

    feats = build_features(df, scale, hm, am)
    feats.to_csv(ROOT / 'loc' / 'exp_new_features_all.csv', index=False, encoding='utf-8-sig')
    print(f'Feature: {feats.shape[0]:,} trận × {feats.shape[1]} cột')

    # kiểm tra Elo có khớp pipeline gốc không
    ref = pd.read_csv(ROOT / 'loc' / 'matches_truoc_wc_2026_elo_features.csv')
    ref['date'] = pd.to_datetime(ref['date'])
    merged = ref.merge(feats, on=['date', 'home_team', 'away_team'], suffixes=('_ref', '_new'))
    if len(merged):
        gap = (merged['home_elo_pre_ref'] - merged['home_elo_pre_new']).abs().max()
        print(f'Khớp Elo với pipeline gốc: {len(merged):,}/{len(ref):,} trận, lệch lớn nhất {gap:.6f}')
        fgap = (merged['home_form_goals_scored_last5_ref']
                - merged['home_form_goals_scored_last5_new']).abs().max()
        print(f'Lệch lớn nhất của form last5: {fgap:.6f}')

    results, per = {}, {}
    sets = {'baseline (7)': BASE}
    for name, cols in GROUPS.items():
        sets[f'+{name}'] = BASE + cols
    all_new = sorted({c for cols in GROUPS.values() for c in cols})
    sets['+tất cả'] = BASE + all_new
    sets['+h2h+ppg+momentum'] = BASE + GROUPS['h2h'] + GROUPS['points_per_game'] + GROUPS['elo_momentum']
    sets['+h2h+ppg'] = BASE + GROUPS['h2h'] + GROUPS['points_per_game']
    sets['ablation: chỉ Elo (3)'] = ['home_elo_pre', 'away_elo_pre', 'neutral']
    sets['ablation: Elo+diff (4)'] = ['home_elo_pre', 'away_elo_pre', 'neutral', 'elo_diff']
    sets['ablation: Elo+h2h+ppg (8)'] = ['home_elo_pre', 'away_elo_pre', 'neutral',
                                        'h2h_n', 'h2h_gd', 'h2h_ppg',
                                        'home_ppg5', 'away_ppg5']
    sets['7 feat: thay form bằng ppg'] = ['home_elo_pre', 'away_elo_pre', 'neutral',
                                         'home_ppg5', 'away_ppg5',
                                         'home_form_goals_conceded_last5',
                                         'away_form_goals_conceded_last5']
    sets['7 feat: thay form bằng adj'] = ['home_elo_pre', 'away_elo_pre', 'neutral',
                                         'home_adj_scored', 'home_adj_conceded',
                                         'away_adj_scored', 'away_adj_conceded']

    for name, cols in sets.items():
        pm = run_experiment(feats, cols)
        per[name] = pm
        results[name] = summarize(pm)
        results[name]['n_feat'] = len(cols)

    # tương quan giữa feature mới và Elo — giải thích vì sao ít cải thiện
    sample = feats[(feats['date'] >= TRAIN_START) & (feats['date'] < WC_START)].copy()
    for col in ('h2h_gd', 'home_ppg5', 'away_ppg5', 'home_elo_d365',
                'home_adj_scored', 'home_career_matches', 'home_opp_elo5'):
        if col in sample:
            sample[f'{col}_diff'] = sample[col] - sample[col.replace('home', 'away')] \
                if col.startswith('home') and col.replace('home', 'away') in sample else sample[col]
    corr_cols = ['elo_diff', 'h2h_gd', 'home_ppg5_diff', 'home_elo_d365_diff',
                 'home_adj_scored_diff', 'home_career_matches_diff', 'home_opp_elo5_diff']
    corr_cols = [c for c in corr_cols if c in sample]
    print('\nTương quan Pearson với elo_diff (trận 2018-2026):')
    print(sample[corr_cols].corr()['elo_diff'].drop('elo_diff').round(3).to_string())

    base_pm = per['baseline (7)']
    table = pd.DataFrame(results).T
    table = table[['n_feat', 'RPS', 'RPS_pois', 'LogLoss', '1X2', '1X2_pois', 'Exact', 'MAE']]
    table['ΔRPS'] = table['RPS'] - table.loc['baseline (7)', 'RPS']
    table['P(tốt hơn)'] = [
        bootstrap_delta(base_pm, per[name])[3] if name != 'baseline (7)' else np.nan
        for name in table.index
    ]
    table['Giải tốt hơn'] = [
        int((per[name].groupby('tournament')['rps'].mean()
             < base_pm.groupby('tournament')['rps'].mean()).sum())
        if name != 'baseline (7)' else np.nan
        for name in table.index
    ]
    table = table.sort_values('RPS')
    print('\n' + '=' * 96)
    print(f'Walk-forward {len(base_pm)} trận / {base_pm["tournament"].nunique()} giải')
    print('=' * 96)
    print(table.round(4).to_string())

    table.to_csv(ROOT / 'loc' / 'exp_new_features.csv', encoding='utf-8-sig')
    print(f'\nĐã lưu: loc/exp_new_features.csv')

    # --- Quét regularization: bộ nhiều feature cần C nhỏ hơn để không overfit ---
    print('\nQuét C (regularization) cho baseline và bộ nhiều feature:')
    sweep = {}
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        row = {'C': C}
        for label, cols in (('baseline', BASE), ('+tất cả', sets['+tất cả'])):
            pm = run_experiment(feats, cols, C=C)
            row[label] = summarize(pm)['RPS']
        sweep[C] = row
    print(pd.DataFrame(sweep).T.round(4).to_string())

    # Riêng 104 trận WC 2026 — tập mà dự án thật sự quan tâm
    print('\nChỉ trên WC 2026 (104 trận):')
    wc_only = pd.DataFrame({
        name: {k: v for k, v in summarize(pm[pm['tournament'] == 'WC 2026']).items()}
        for name, pm in per.items()
    }).T[['RPS', 'RPS_pois', '1X2', '1X2_pois', 'Exact', 'MAE']]
    print(wc_only.sort_values('RPS').round(4).to_string())

    best = table.index[0]
    if best != 'baseline (7)':
        print(f'\nSo sánh theo từng giải: {best} vs baseline (RPS)')
        a = base_pm.groupby('tournament')['rps'].mean()
        b = per[best].groupby('tournament')['rps'].mean()
        cmp = pd.DataFrame({'baseline': a, best: b})
        cmp['Δ'] = cmp[best] - cmp['baseline']
        print(cmp.round(4).to_string())


if __name__ == '__main__':
    main()
