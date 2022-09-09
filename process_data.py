import pandas as pd

df20 = pd.read_csv('2020.csv')
df20['isTie'] = df20['PtsW']==df20['PtsL']
df20['Week'] = '2020-'+df20['Week']

df21 = pd.read_csv('2021.csv')
df21['isTie'] = df21['PtsW']==df21['PtsL']
df20['Week'] = '2021-'+df21['Week']

df = pd.concat([df20, df21]).reset_index()
df = df[['Week', 'Date', 'Winner/tie', 'Loser/tie', 'isTie', 'PtsW', 'PtsL']]
df.to_csv('results.csv')