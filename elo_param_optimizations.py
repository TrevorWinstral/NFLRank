import pandas as pd
import numpy as np
df = pd.read_csv('results.csv', index_col=0)
weeks = ['0']
weeks.extend(list(df['Week'].unique()))
teams = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders']
team_ind = {teams[i]:i for i in range(len(teams))}

base_elo = 0
realization_delta = pd.DataFrame(index=teams)
reset = lambda factor, level: factor*(level- base_elo) + base_elo # for bracket reset, see more below

def score_realization_diff(row, expected_score_matrix=None):
    #assert expected_score_matrix != None
    global realization_delta
    t1,t2 = row[['Winner/tie', 'Loser/tie']]
    t1i, t2i = team_ind[t1], team_ind[t2]
    expected_scores = [expected_score_matrix[t1i, t2i], expected_score_matrix[t2i, t1i]] #elo diff between winner and loser
    #realized score for winner t1 = 1, vs 0 for loser t2
    if row['isTie']:
        realization_delta.loc[t1, 'delta'] += 0.5-expected_scores[0]
        realization_delta.loc[t2, 'delta'] += 0.5-expected_scores[1]
    else:
        realization_delta.loc[t1, 'delta'] += 1-expected_scores[0]
        realization_delta.loc[t2, 'delta'] += 0-expected_scores[1]

def elo_sim(baseK, logistic_factor, reset_factor):
    global realization_delta
    realization_delta = pd.DataFrame(index=teams)

    elo = pd.DataFrame(index=teams)
    elo['0'] = [base_elo]*32 #elo[week] tells you elo after that week has concluded
    K = baseK
    for w_ind in range(1, len(weeks)):
        realization_delta['delta'] = 0
        w = weeks[w_ind] # current week
        last_w = weeks[w_ind-1] # last week
        games = df[df['Week']==w] # games for week w
        #calculate expected score for winner
        ones = np.array([1]*32)
        if w[:4] != last_w[:4] and last_w[0] == '2':
            #print('resetting')
            elo['Reset Week ' + w[:4]] = reset(reset_factor, elo[last_w])
            last_w = 'Reset Week ' + w[:4] # last week set to reset week
            K = baseK
        elo_diffs = np.outer(elo[last_w],ones) - np.outer(elo[last_w],ones).T
        expected_score_matrix = 1/(1 + 10**(-1*elo_diffs/logistic_factor))# expected score for i playing against j given by [i,j]'th entry

        # adjust for realized score
        _=games.apply(score_realization_diff, args=(expected_score_matrix,), axis=1)
        realization_delta['delta'] *= K
        elo[w] = elo[last_w] + realization_delta['delta']
        K -= 0 #1/3
    return elo

logistic_factor = 100 # 400 for chess, the larger the more likely upsets are, so upsetters are rewarded less
reset_factor = 0.65 #after season do factor * (elo-base_elo) + base_elo, 0 is hard reset, 1 is no reset
baseK = 40 #maximal change of elo per game (this maybe should adjust throughout the season, probably declining)
elo = elo_sim(baseK, logistic_factor, reset_factor)


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# We have 3 main arbitrary parameters, logistic factor, K (and the amount that we change K by), and the reset factor
# we want to minimize MSE between between expected score and realized score. We could just find min over all combinations of these
actual_score_matrix = np.zeros((32,32))
def apply_scores(row):
    t1, t2 = row[['Winner/tie', 'Loser/tie']]
    t1i, t2i = team_ind[t1], team_ind[t2]
    if row['isTie']:
        actual_score_matrix[t1i, t2i] = 0.5
        actual_score_matrix[t2i, t1i] = 0.5
    else:
        actual_score_matrix[t1i, t2i] = 1
        actual_score_matrix[t2i, t1i] = 0

def elo_MSE(elo, results):
    global actual_score_matrix
    SSE = 0

    for w_ind in range(1, len(weeks)):
        realization_delta['delta'] = 0
        w = weeks[w_ind] # current week
        last_w = weeks[w_ind-1] # last week
        g = results[results['Week']==w] # games for week w
        #calculate expected score for winner
        ones = np.array([1]*32)
        if w[:4] != last_w[:4] and last_w[0] == '2':
            last_w = 'Reset Week ' + w[:4] # last week set to reset week
        elo_diffs = np.outer(elo[last_w],ones) - np.outer(elo[last_w],ones).T
        expected_score_matrix = 1/(1 + 10**(-1*elo_diffs/logistic_factor))# expected score for i playing against j given by [i,j]'th entry
        actual_score_matrix = np.copy(expected_score_matrix)
        _ = g.apply(apply_scores, axis=1)
        scores_half = np.triu(actual_score_matrix - expected_score_matrix, 1)**2 # we don't need to count games twice
        SSE += np.sum(scores_half)

    return SSE/(len(weeks)-1)

KN, LN, RN =  31, 16, 21 #31, 16, 21
K_range = np.linspace(10,40,KN)
logistic_range = np.linspace(50, 800, LN)
reset_range = np.linspace(0, 1, RN)

from tqdm.auto import tqdm
results = np.zeros((KN, LN, RN))
for k in tqdm(range(len(K_range))):
    for l in tqdm(range(len(logistic_range))):
        for r in range(len(reset_range)):
            e = elo_sim(K_range[k],logistic_range[l],reset_range[r])
            results[k,l,r] = elo_MSE(e, df)

flat_ind=np.argmin(results)
unflat_ind = np.unravel_index(flat_ind, results.shape)
k_opt = K_range[unflat_ind[0]]
l_opt = logistic_range[unflat_ind[1]]
r_opt = reset_range[unflat_ind[2]]
print(k_opt, l_opt, r_opt) # 11.0, 200.0, 0.7000000000000001
#k_opt, l_opt, r_opt=11.0, 200.0, 0.70

plt.hist(results.flatten())
plt.show()

elo = elo_sim(k_opt, l_opt, r_opt)
f = plt.figure()
for i in range(16):
    _=elo.loc[teams[i]].plot(label=teams[i])

f.set_figwidth(5)
f.legend(loc='upper left')
f.show()
f = plt.figure()
for i in range(16,32):
    _=elo.loc[teams[i]].plot(label=teams[i])

f.set_figwidth(5)
f.legend(loc='upper left')
f.show()

w = weeks[-1]
best_teams=elo[w].sort_values(ascending=False)
top12 = list(best_teams.index[:12])
f = plt.figure()
for i in top12:
    _=elo.loc[i,weeks].plot(label=i)

f.set_figwidth(5)
plt.axvline(x=21, color='black', linestyle='--')
f.legend(loc='upper left')
f.show()
