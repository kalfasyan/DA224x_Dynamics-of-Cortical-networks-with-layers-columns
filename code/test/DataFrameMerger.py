import os
import pandas as pd

fname = '000'

path = os.getcwd()
files = os.listdir(path)
files_xls = [f for f in files if f[:3] == fname]
df = pd.DataFrame()

for f in files_xls:
    data = pd.read_excel(f,'Sheet1')
    df = df.append(data)

df.sort_index(inplace=True)
df = df.sort(['from_to'])
df.to_excel('Pandas'+fname+'.xlsx')
print "Done!"