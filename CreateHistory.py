import pandas as pd
import numpy as np

data=pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\GeorgiaPac.pkl")
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







def createHistoryOneLevel(column,data_frame,test_data_frame,values):
        grp_column=[]
        data_frame['temp']=data_frame[column]
        test_data_frame['temp_test']=test_data_frame[column]



        print(column)
        data_frame[column].replace('\\N', np.nan, inplace=True)
        data_frame[column].replace(np.nan, 'null', inplace=True)
        data_frame[column] = data_frame[column].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
        data_frame[column] = data_frame[column].astype(str)
        #data_frame[column] = data_frame.column.astype(str)
        print(data_frame[column].dtype)
        grp_column = data_frame.groupby(column).agg({'output': 'sum', 'pk_deduction_id': 'count'})
        grp_column.reset_index(inplace=True)
        grp_column['output'] = grp_column.pk_deduction_id - grp_column.output
        grp_column.columns = [column, 'invalid', 'total']

        grp_column.sort_values(by='total', ascending=False, inplace=True)


        grp_column.sort_values(by='total', ascending=False, inplace=True)
        grp_column.loc[grp_column['total'] <= int(values), column] = 'others'
        grp_column[column] = grp_column[column].astype('str')


        grp_column_list=grp_column[column].unique().tolist()
        print(grp_column_list)

        data_frame[column] = data_frame[column].apply(lambda x: x if (x in grp_column_list) else 'others')
        data_frame.reset_index(inplace=True)
        data_frame.drop('index', axis=1, inplace=True)
        
        #calculating history
        output_list = data_frame.output.values
        from sklearn.model_selection import StratifiedKFold
        stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
        stf.get_n_splits(data_frame,output_list)
        data_frame[str(column+"_history")] = np.nan

        for x_four_index, x_one_index in stf.split(data_frame,output_list):
            x_four, x_one = data_frame.iloc[x_four_index], data_frame.iloc[x_one_index]
            print('x_four : ', x_four.shape, 'x_one : ', x_one.shape)
            x_four['output1'] = x_four.output.map({1: 0, 0: 1})
            hstry_val = x_one[column].map(x_four.groupby(column).output1.mean())
            x_one['mean_hstry'] = hstry_val
            data_frame.loc[x_one_index, str(column+"_history")] = x_one['mean_hstry']

        #merging with test
        grp_column = grp_column.groupby(column).agg({'invalid': 'sum', 'total': 'sum'}).reset_index()
        grp_column[str(column+"_history")] = grp_column['invalid'] / grp_column['total']

        #test processing
        test_data_frame[column].replace('\\N', np.nan, inplace=True)
        test_data_frame[column].replace(np.nan, 'null', inplace=True)
        test_data_frame[column] = test_data_frame[column].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
        test_data_frame[column] = test_data_frame[column].astype(str)

        test_data_frame[column] = test_data_frame[column].apply(lambda x: x if (x in grp_column_list) else 'others')
        test_data_frame = pd.merge(test_data_frame, grp_column, on=column, how='left')


        del data_frame[column]
        del test_data_frame[column]

        data_frame.rename(columns={'temp':column},inplace=True)
        test_data_frame.rename(columns={'temp_test': column}, inplace=True)
        #print(test_data_frame)



        return data_frame,test_data_frame



train,test=createHistoryOneLevel('payer',train,test,'80')
print(train['payer'].value_counts(dropna=False))
#train,test=createHistoryOneLevel('ar_reason_code',train,test,'80')

#print(train['payer_history'])
#print(test['payer_history'])

list=train['payer_history'].tolist()
print(train['payer'].nunique())
print(train['payer_history'].nunique())
print("---------------------------------")
#df=train['pk_deduction_id','payer','payer_history']
print(len(list))