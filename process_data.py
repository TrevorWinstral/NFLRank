import pandas as pd
# data from https://www.pro-football-reference.com/years/2021/games.htm
df20 = pd.read_csv('2020.csv')
df20['isTie'] = df20['PtsW']==df20['PtsL']
df20['Week'] = '2020-'+df20['Week']

df21 = pd.read_csv('2021.csv')
df21['isTie'] = df21['PtsW']==df21['PtsL']
df21['Week'] = '2021-'+df21['Week']

df22 = pd.read_csv('2022.csv')
df22['isTie'] = df22['PtsW']==df22['PtsL']
df22['Week'] = '2022-'+df22['Week'].astype(str)

df = pd.concat([df20, df21, df22]).reset_index()
df = df[['Week', 'Date', 'Winner/tie', 'Loser/tie', 'isTie', 'PtsW', 'PtsL', 'type']]
df[df['type']=='boxscore'].to_csv('results.csv')

#get next weeks games
next_week=df[df['type']=='preview'].iloc[0]['Week']
df[df['Week'] == next_week].to_csv('next_games')