import os
import pandas as pd

def gen_readlist(x):
    x = x.sort_values(by='Timestamp')
    return list(zip(x['MovieID'], x['Rating'], x['Timestamp']))
    


basedir = '/Users/zhanghaoyang/Desktop/Movie_Recsys/'

rating_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
rating_data = pd.read_csv(os.path.join(basedir, 'data', 'ratings.dat'),
                          names=rating_columns,
                          delimiter='::')

readlist = rating_data.groupby('UserID').apply(gen_readlist).to_dict()
print(readlist[1])