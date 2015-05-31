import os
import pandas as pd
import glob

fname = '000'

path = os.getcwd()
files = os.listdir(path)
files = glob.glob("*"+fname+"*")

df1,df2,df3 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

df1 = pd.read_excel(files[0])
df2 = pd.read_excel(files[1])
df3 = pd.read_excel(files[2])

ez =  pd.concat([df1,df2,df3], axis=1)

ez.to_excel("Pandas"+fname+"FINAL.xlsx")
print "Done!"