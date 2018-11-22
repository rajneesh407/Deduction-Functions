import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data=pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\GP\three_tables_4_lakhs.pkl")
data=data.drop_duplicates('pk_deduction_id')
data=data.dropna(axis=0,subset=['fk_acct_doc_header_id'])
data.reset_index(inplace=True,drop=True)

data['fk_action_code_id']=data['fk_action_code_id'].fillna(-1)
data.fk_action_code_id.value_counts(dropna=False)
data['fk_action_code_id']=data['fk_action_code_id'].astype(int)
listid=[341,419,422,425,240,243,246,428,300]
data['output'] = np.nan
data.loc[(data.fk_action_code_id.isin(listid)), 'output'] = 0
data.output.fillna(value=1, inplace=True)
data['output']=data.output.astype(int)


import datetime
train = data.loc[(data.deduction_created_date >= datetime.datetime(2015,1,2)) & (data.deduction_created_date <= datetime.datetime(2018,5,1))]
#print(train.shape)
#print(train)
test = data.loc[data.deduction_created_date >= datetime.datetime(2018, 1, 2)]
#print(test.shape)
train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)



