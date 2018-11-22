import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

data = pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\GP\three_tables_4_lakhs.pkl")

data = data.drop_duplicates('pk_deduction_id')
data.reset_index(inplace=True, drop=True)

data=data.dropna(axis=0,subset=['fk_acct_doc_header_id'])
data.reset_index(inplace=True, drop=True)

data['fk_action_code_id'] = data['fk_action_code_id'].fillna(-1)
data.fk_action_code_id.value_counts(dropna=False)
data['fk_action_code_id'] = data['fk_action_code_id'].astype(int)
listid = [341, 419, 422, 425, 240, 243, 246, 428, 300]
data['output'] = np.nan
data.loc[(data.fk_action_code_id.isin(listid)), 'output'] = 0
data.output.fillna(value=1, inplace=True)
data['output'] = data.output.astype(int)







def dataProcessing(data):
    # Replacing \N with np.nan`
    data = data.replace('\\N', np.nan)

    # Replacing None with np.nan
    data = data.fillna(value=np.nan)

    # Replacing the blank values
    data = data.replace(r'^\s*$', np.nan, regex=True)

    return data

def trainTestSplit(data):
    import datetime
    import datetime

    train = data.loc[(data.deduction_created_date >= datetime.datetime(2015, 1, 2))&(data.deduction_created_date <= datetime.datetime(2018, 5, 1))]
    # print(train.shape)
    # print(train)
    test = data.loc[data.deduction_created_date > datetime.datetime(2018, 1, 2)]

    print(train.shape)
    print(test.shape)
    return  train,test

def get_summary(column_list):
    # Creating the summary dataframe
    summary_df = pd.DataFrame(columns=['column_name', 'null_percentage', 'no_of_uniques', 'value_counts'])
    summary_df['column_name'] = column_list

    # Creating summary
    for col in column_list:
        # null percentage
        summary_df.loc[summary_df.column_name == col, 'null_percentage'] = str(
            np.round((df[col].isnull().sum() / df.shape[0]) * 100, 2)) + ' % '

        # number of uniques
        summary_df.loc[summary_df.column_name == col, 'no_of_uniques'] = df[col].nunique()

        # value_counts

        ## Creating the value_counts dataframe
        vc = pd.DataFrame(df[col].value_counts(dropna=False).head(5)).reset_index()
        vc.columns = [col, 'count']
        vc['ratio'] = np.round(vc['count'] / len(df), decimals=2) * 100

        ## saving the dataframe in string list
        l = []
        for k in zip(vc[col].tolist(), vc['count'].tolist(), vc['ratio'].tolist()):
            l.append(str(k[0]) + ' --> ' + str(k[1]) + ' --> ' + str(k[2]) + ' % ')

        ## merging the string list
        s = ''
        for val in l:
            s = s + str(val) + ' || '

        # [:-3] to remove last || and space
        summary_df.loc[summary_df.column_name == col, 'value_counts'] = s[:-3]

    return summary_df


def startOneLevelHistoryList(data,train,test):
    summary=get_summary(data.columns.tolist())

    x=summary.loc[(summary['no_of_uniques']>3 )& (summary['no_of_uniques']<(data.shape[0]/4) )]['column_name'].tolist()
    return x


data=dataProcessing(data)
df=data
train,test=trainTestSplit(data)
one_level_list=startOneLevelHistoryList(data,train,test)