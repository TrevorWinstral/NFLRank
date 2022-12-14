from cmath import log
from tkinter import CURRENT
import pandas as pd
import numpy as np
CURRENT_SEASON='2022'

df = pd.read_csv('results.csv', index_col=0)
weeks = ['0']
weeks.extend(list(df['Week'].unique()))
teams = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders']
team_ind = {teams[i]:i for i in range(len(teams))}

base_elo = 0
realization_delta = pd.DataFrame(index=teams)
reset = lambda factor, level: factor*(level- base_elo) + base_elo # for bracket reset, see more below

#k_opt, l_opt, r_opt= 15.0 125.0 0.7  
logistic_factor = 125 # 400 for chess, the larger the more likely upsets are, so upsetters are rewarded less
reset_factor = 0.7 #after season do factor * (elo-base_elo) + base_elo, 0 is hard reset, 1 is no reset
baseK = 15 #maximal change of elo per game (this maybe should adjust throughout the season, probably declining)

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

elo = elo_sim(baseK, logistic_factor, reset_factor)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
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
step=21
for s in range(1,6):
    plt.axvline(x=s*step, color='black', linestyle='--')
f.legend(loc='upper left')
f.show()


def predict(row, elos):
    t1 = row['Winner/tie']
    t2 = row['Loser/tie']

    elo_diff = elos[t1] - elos[t2]
    expected_score = 1/(1 + 10**(-1*elo_diff/logistic_factor))

    return expected_score
    

gs = pd.read_csv('next_games.csv')[['Winner/tie', 'Loser/tie']]
elos = elo.iloc[:,-1]
gs['Prediction'] = gs.apply(predict, args=(elos,), axis=1)
gs.rename(columns={'Winner/tie':'Team 1', 'Loser/tie':'Team 2'}, inplace=True)
gs
print(gs.to_html(index=False))
gs.to_html('predictions.html', index=False)


# create current power rankings
power_rankings=pd.DataFrame(np.array(best_teams.index).reshape((8,4)))
print(power_rankings.to_html(index=False, header=False))
power_rankings.to_html('power_rankings.html', index=False, header=False)


# power rankings for this season
current_season = [w for w in elo.columns if w[:4]==CURRENT_SEASON or w==f'Reset Week {CURRENT_SEASON}']
current_season.reverse()
rankings = pd.DataFrame([elo[w].sort_values(ascending=False).index for w in current_season]).T
new_colnames=['Week 0']
new_colnames.extend([f'Week {w[5:]}' for w in elo.columns if w[:4]==CURRENT_SEASON])
new_colnames.reverse()
rankings.columns= new_colnames
print(rankings.to_html())
rankings.to_html('historical_rankings.html')

# Historical Prediction Accuracy
def logistical_scores(row):
    pts1 = row['PtsW']
    pts2 = row['PtsL']
    score = 1/(1 + 10**(-0.1*(pts1-pts2)))
    return score

def predict_total(row, elo):
    week = row['Week']
    if week == '0':
        return 0.5
    last_week_ind = week_to_ind[week]-1
    t1 = row['Winner/tie']
    t2 = row['Loser/tie']
    t1i, t2i = team_ind[t1], team_ind[t2]
    elo_diff = elo.iloc[t1i, last_week_ind] - elo.iloc[t2i, last_week_ind]
    prediction = 1/(1 + 10**(-1*elo_diff/logistic_factor))
    return prediction

all_weeks = list(df['Week'].unique())
week_to_ind = {all_weeks[i]:i for i in range(len(all_weeks))}
df['Normalized Score'] = df.apply(logistical_scores, axis=1)
df.apply(predict_total, axis=1, args=(elo,))
df['Prediction'] = df.apply(predict_total, axis=1, args=(elo,))
df['Game Score'] = df['PtsW'].astype(int).astype(str) + '-' + df['PtsL'].astype(int).astype(str)
df[['Week', 'Winner/tie', 'Loser/tie', 'Game Score', 'Prediction', 'Normalized Score']]

df['Error'] = df['Normalized Score'] - df['Prediction'] # How wrong was I
df['Qualitative Error'] = (df['Prediction']>=0.5) # Did I guess the right winner
df[['Week', 'Winner/tie', 'Loser/tie', 'Game Score', 'Prediction', 'Normalized Score', 'Qualitative Error']].tail(20)
def correctness(row):
    bad=':-1:' # thumbs down on github
    good=':+1:' # thumbs up on github
    # 1 thumbs up if correct, 2 if within 7 points, 3 if within 3 points
    # 1 thumbs down if within 3, 2 if within 7, else 3
    # first transform predicted score back into actual point differential
    predicted_diff = np.log10((1/row['Prediction'])-1)/(-0.1)
    realized_diff = np.log10((1/row['Normalized Score'])-1)/(-0.1)
    diff_diff = abs(realized_diff - predicted_diff)
    if row['Qualitative Error']:
        output = good + (diff_diff<=7)*good + (diff_diff<=3)*good
    else:
        output = (3 - (diff_diff<=7) - (diff_diff<= 3))*bad
    return output


s = df.loc[df['Week'].str[:4] == CURRENT_SEASON][['Date', 'Winner/tie', 'Loser/tie', 'Game Score', 'Normalized Score', 'Prediction', 'Qualitative Error', 'Error']]
s = s.reset_index(drop=True).iloc[::-1]
s['Accuracy'] = s.apply(correctness, axis=1)
s['Normalized Score'] = s['Normalized Score'].round(decimals=3)
s['Prediction'] = s['Prediction'].round(decimals=3)
s['Error'] = s['Error'].round(decimals=3)
s.to_html('historical_performance.html', index=False, columns=['Date', 'Winner/tie', 'Loser/tie', 'Game Score', 'Accuracy', 'Normalized Score', 'Prediction'])
num_games = s.shape[0]
num_correct = s['Qualitative Error'].sum()
print(f'Prediction Record {num_correct}-{num_games-num_correct} ({(num_correct/num_games)*100:0.1f}%)')
