from pandas import read_csv
from datetime import datetime
import numpy as np


# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d')


dataset = read_csv(r'C:\Users\T\OneDrive\My Research\LTLF\Program\EUNITE_Data.csv', delimiter=',',
                   parse_dates=[['Year', 'Month', 'Day']], index_col=0, date_parser=parse, keep_date_col=True)

print(dataset.columns.values.tolist())
dataset = dataset[['MaxLoad', 'temp']]
print(dataset.shape)
print(dataset.columns.values.tolist())

# save to file
dataset.to_csv('EUNITE_Load.csv')
