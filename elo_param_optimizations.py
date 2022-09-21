from cmath import log
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


# I should import these, but for now I just have to copy them
def score_realization_diff(row, expected_score_matrix=None):
    #assert expected_score_matrix != None
    global realization_delta
    t1,t2 = row[['Winner/tie', 'Loser/tie']]
    t1i, t2i = team_ind[t1], team_ind[t2]
    expected_scores = [expected_score_matrix[t1i, t2i], expected_score_matrix[t2i, t1i]] #elo diff between winner and loser
    #realized score for winner t1 = 1, vs 0 for loser t2
    ## Old way without logistic score curve
    # if row['isTie']:
    #     realization_delta.loc[t1, 'delta'] += 0.5-expected_scores[0]
    #     realization_delta.loc[t2, 'delta'] += 0.5-expected_scores[1]
    # else:
    #     realization_delta.loc[t1, 'delta'] += 1-expected_scores[0]
    #     realization_delta.loc[t2, 'delta'] += 0-expected_scores[1]

    ## New way with logistic curve
    realized_point_diff = row['PtsW'] - row['PtsL']
    logistic_point_diff = 1/(1+10**(realized_point_diff*-0.1))
    realization_delta.loc[t1, 'delta'] += logistic_point_diff-expected_scores[0]
    realization_delta.loc[t2, 'delta'] += (1-logistic_point_diff)-expected_scores[1]

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
            elo = elo.copy() # defragment frame for performance
            K = baseK
        elo_diffs = np.outer(elo[last_w],ones) - np.outer(elo[last_w],ones).T
        expected_score_matrix = 1/(1 + 10**(-1*elo_diffs/logistic_factor))# expected score for i playing against j given by [i,j]'th entry

        # adjust for realized score
        _=games.apply(score_realization_diff, args=(expected_score_matrix,), axis=1)
        realization_delta['delta'] *= K
        elo[w] = elo[last_w] + realization_delta['delta']
        K -= 0 #1/3
    return elo


import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# We have 3 main arbitrary parameters, logistic factor, K (and the amount that we change K by), and the reset factor
# we want to minimize MSE between between expected score and realized score. We could just find min over all combinations of these
actual_score_matrix = np.zeros((32,32))
def apply_scores(row):
    global actual_score_matrix
    t1, t2 = row[['Winner/tie', 'Loser/tie']]
    t1i, t2i = team_ind[t1], team_ind[t2]
    ## Old way without logistic score diff
    # if row['isTie']:
    #     actual_score_matrix[t1i, t2i] = 0.5
    #     actual_score_matrix[t2i, t1i] = 0.5
    # else:
    #     actual_score_matrix[t1i, t2i] = 1
    #     actual_score_matrix[t2i, t1i] = 0
    ## Logistic score diff
    realized_point_diff = row['PtsW'] - row['PtsL']
    logistic_point_diff = 1/(1+10**(-0.1*realized_point_diff))
    actual_score_matrix[t1i, t2i] = logistic_point_diff
    actual_score_matrix[t2i, t1i] = 1-logistic_point_diff

def elo_MSE(elo, results, logistic_factor):
    global actual_score_matrix
    SSE = 0

    for w_ind in range(1, len(weeks)):
        realization_delta['delta'] = 0
        w = weeks[w_ind] # current week
        last_w = weeks[w_ind-1] # last week
        g = results[results['Week']==w] # games for week w
        #g = df[df['Week']==w] # games for week w
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

    return SSE/len(results)


# Original search used following and found optimal k,l,r = 38.0, 350.0, 0.675  
# KN, LN, RN =  31, 16, 21 #, and the spaces
# K_range = np.linspace(10,40,KN)
# logistic_range = np.linspace(50, 800, LN)
# reset_range = np.linspace(0.5, 1, RN) 
# Now we refine around these values
# KN, LN, RN =  3, 6, 26
# K_range = np.linspace(10, 12,KN)
# logistic_range = np.linspace(100, 300, LN)
# reset_range = np.linspace(0.5, 0.75, RN)

# ## I want to try one singlular optimization.
# KN, LN, RN =  10, 17, 13
# K_range = np.linspace(5,50, KN) # 5-50 stepsize 5
# logistic_range = np.linspace(100, 500, LN) # 100-500 stepsize 25
# reset_range = np.linspace(0.25, 0.85, RN) # 0.25-0.85 stepsize 0.05
KN, LN, RN =  9, 17, 13
K_range = np.linspace(10,50, KN) # 10-50 stepsize 5
logistic_range = np.linspace(100, 500, LN) # 100-500 stepsize 25
reset_range = np.linspace(0.25, 0.85, RN) # 0.25-0.85 stepsize 0.05

from tqdm.auto import tqdm
results = np.zeros((KN, LN, RN))
for k in tqdm(range(len(K_range))):
    for l in tqdm(range(len(logistic_range)), leave=False):
        for r in tqdm(range(len(reset_range)), leave=False, desc=f'k={K_range[k]}, l={logistic_range[l]}'):
            print(' ', K_range[k],logistic_range[l],reset_range[r])
            e = elo_sim(K_range[k],logistic_range[l],reset_range[r])
            results[k,l,r] = elo_MSE(e, df, logistic_range[l])

flat_ind=np.argmin(results)
unflat_ind = np.unravel_index(flat_ind, results.shape)
k_opt = K_range[unflat_ind[0]]
l_opt = logistic_range[unflat_ind[1]]
r_opt = reset_range[unflat_ind[2]]
print(k_opt, l_opt, r_opt) # 15.0 125.0 0.7  
#k_opt, l_opt, r_opt=11.0, 200.0, 0.70 # SPARSE 
#k_opt, l_opt, r_opt=11.0, 100.0, 0.68, # REFINED

plt.hist(results.flatten())
plt.show()

# elo = elo_sim(k_opt, l_opt, r_opt)
# f = plt.figure()
# for i in range(16):
#     _=elo.loc[teams[i]].plot(label=teams[i])


