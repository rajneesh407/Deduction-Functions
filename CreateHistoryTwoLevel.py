import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data=pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\three_tables_4_lakhs.pkl")
data=data.drop_duplicates('pk_deduction_id')

data['fk_action_code_id']=data['fk_action_code_id'].fillna(-1)
data.fk_action_code_id.value_counts(dropna=False)
data['fk_action_code_id']=data['fk_action_code_id'].astype(int)
listid=[341,419,422,425,240,243,246,428,300]
data['output'] = np.nan
data.loc[(data.fk_action_code_id.isin(listid)), 'output'] = 0
data.output.fillna(value=1, inplace=True)
data['output']=data.output.astype(int)


import datetime
train = data[(data.deduction_created_date >= datetime.datetime(2015,1,2)) & (data.deduction_created_date <= datetime.datetime(2018,5,1))]
print(train.shape)
test = data[data.deduction_created_date > datetime.datetime(2018, 1, 2)]
print(test.shape)



















def CreateHistoryTwoLevel(column1,column2,train_data_frame,test_Data_frame,value):
 ## creating regularised history
 from sklearn.model_selection import StratifiedKFold
 print(column1)
 train_data_frame[column1].replace('\\N', np.nan, inplace=True)
 train_data_frame[column1].replace(np.nan, 'null', inplace=True)
 train_data_frame[column1] = train_data_frame[column1].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
 train_data_frame[column1] = train_data_frame[column1].astype(str)

 train_data_frame[column2].replace('\\N', np.nan, inplace=True)
 train_data_frame[column2].replace(np.nan, 'null', inplace=True)
 train_data_frame[column2] = train_data_frame[column2].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
 train_data_frame[column1] = train_data_frame[column2].astype(str)

 stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
 output_list = train_data_frame.output.values

 stf.get_n_splits(train_data_frame, output_list)

 #stf = StratifiedKFold(output_list, n_folds=5, shuffle=True, random_state=8)
 train_data_frame[column1+"_"+column2+"_history"] = np.nan
 for x_four_index, x_one_index in stf.split(train_data_frame, output_list):
    x_four, x_one = train_data_frame.iloc[x_four_index], train_data_frame.iloc[x_one_index]
    x_four = x_four[['pk_deduction_id', column1, column2, 'output']]
    x_one = x_one[['pk_deduction_id', column1, column2, 'output']]
    print(x_four.shape, x_one.shape)

    # To know the total count of every pair
    grp_reason_code = x_four.groupby([column1, column2]).agg({'output': 'sum', 'pk_deduction_id': 'count'})
    grp_reason_code.reset_index(inplace=True)
    grp_reason_code['output'] = grp_reason_code['pk_deduction_id'] - grp_reason_code['output']
    grp_reason_code.rename(columns={'output': 'invalid', 'pk_deduction_id': 'total'}, inplace=True)
    grp_reason_code.sort_values([column1, 'total'], ascending=[True, False], inplace=True)
    print("line76")
    print(grp_reason_code.shape)
    # Calculating the history
    temp = grp_reason_code.groupby([column1])
    key = list(temp.groups.keys())
    temp_hstry = pd.DataFrame()
    for i in key:
        df = temp.get_group(i)
        df[column2+"_label"] = df[column2]
        df.loc[df.total < int(value), column2+"_label"] = 'others'
        df[column1+column2+"_history"] = df['invalid'] / df['total']

        if len(df[df[column2+"_label"] == 'others']) != 0:
            df.loc[df[column2+"_label"] == 'others', column1+column2+"_history"] = (
                        (df[df[column2+"_label"] == 'others']['invalid'].sum()) / (
                    df[df[column2+"_label"] == 'others']['total'].sum()))

        elif len(df[df[column2+"_label"] == 'others']) == 0:
            df.loc['abc'] = [df[column1].values[0], 'Random', df.invalid.values[0], df.total.values[0], 'others', 0]

        temp_hstry = pd.concat([temp_hstry, df], axis=0)

    temp_hstry = temp_hstry[[column1, column2, column2+"_label", column1+column2+"_history"]]
    temp_hstry.rename(
        columns={column2: column2+"_two", column2+"_label": column2+"_label_two"},
        inplace=True)
    temp_hstry.reset_index(inplace=True)
    temp_hstry.drop('index', axis=1, inplace=True)

    x_one[column2+'_two_level'] = x_one[column2]
    x_one = pd.merge(x_one, temp_hstry, how='left', left_on=[column1, column2+'_two_level'],
                     right_on=[column1, column2+'_two'])
    print("line107")
    print(x_one.shape)
    del x_one[column1+column2+"_history"]

    x_one.loc[x_one[column2+'_label_two'].isnull(), column2+'_label_two'] = 'others'

    x_one = pd.merge(x_one, temp_hstry.drop_duplicates([column1, column2+'_label_two']),
                     left_on=[column1, column2+'_label_two'], right_on=[column1, column2+'_label_two'],
                     how='left')
    x_one.set_index(x_one_index, inplace=True)
    print(x_one[column1+column2+"_history"].isnull().sum())
    #print(x_one)
    print("line 120")
    print(train_data_frame.columns)
    print(x_one.columns)
    print(list(x_one_index))
    train_data_frame.loc[x_one_index, str(column1+"_"+column2+"_history")] = x_one[column1+column2+"_history"]
    return train_data_frame


train.reset_index(inplace=True)
train.reset_index(inplace=True)
test.reset_index(inplace=True)
test.reset_index(inplace=True)

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

train=CreateHistoryTwoLevel('payer','ar_reason_code',train,test,"20")
print(train['payer_ar_reason_code_history'])