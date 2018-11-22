import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data=pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\three_tables_4_lakhs.pkl")
data=data.drop_duplicates('pk_deduction_id')
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
test = data.loc[data.deduction_created_date > datetime.datetime(2018, 1, 2)]
#print(test.shape)










def handlePreprocessing(train_data_frame,test_data_frame,column1,column2):
    train_data_frame[column1].replace('\\N', np.nan, inplace=True)
    train_data_frame[column1].replace(np.nan, 'null', inplace=True)
    train_data_frame[column1] = train_data_frame[column1].apply(
        lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    train_data_frame[column1] = train_data_frame[column1].astype(str)

    train_data_frame[column2].replace('\\N', np.nan, inplace=True)
    train_data_frame[column2].replace(np.nan, 'null', inplace=True)
    train_data_frame[column2] = train_data_frame[column2].apply(
        lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    train_data_frame[column2] = train_data_frame[column2].astype(str)

    # test Data processing
    test_data_frame[column1].replace('\\N', np.nan, inplace=True)
    test_data_frame[column1].replace(np.nan, 'null', inplace=True)
    test_data_frame[column1] = test_data_frame[column1].apply(
        lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_data_frame[column1] = test_data_frame[column1].astype(str)

    test_data_frame[column2].replace('\\N', np.nan, inplace=True)
    test_data_frame[column2].replace(np.nan, 'null', inplace=True)
    test_data_frame[column2] = test_data_frame[column2].apply(
        lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_data_frame[column2] = test_data_frame[column2].astype(str)
    return train_data_frame,test_data_frame

def twoLevelHistory(column1,column2,train_Data_frame,test_data_frame,value):
    # reason code and category level history
    train_Data_frame,test_data_frame=handlePreprocessing(train_Data_frame,test_data_frame,column1,column2)

    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    train_Data_frame[column2] = train_Data_frame[column2].astype(str)

    reason_category = train_Data_frame.groupby([column1, column2]).agg(
        {'output': 'sum', 'pk_deduction_id': 'count'}).sort_values('pk_deduction_id', ascending=False)

    reason_category.reset_index(inplace=True)

    reason_category.groupby([column1, column2]).agg({'output': 'first', 'pk_deduction_id': 'first'})

    reason_category.reset_index(inplace=True)

    del reason_category['index']

    reason_category['output'] = reason_category['pk_deduction_id'] - reason_category['output']
    reason_category.columns = [column1, column2, 'invalid', 'new_total']

    rt_category = train_Data_frame.groupby([column1]).agg({'pk_deduction_id': 'count'})
    rt_category.reset_index(inplace=True)
    rt_category.columns = [column1, 'grand_total_sum']

    final_reason_category = pd.merge(reason_category, rt_category, how='left', on=column1)

    x = final_reason_category.groupby([column1])
    u = list(x.groups.keys())
    final = pd.DataFrame()

    for i in u:
        df = x.get_group(i)
        df['cumsum'] = df['new_total'].cumsum()
        df['grand_total_sum'] = (0.90 * df['grand_total_sum']) - df['cumsum']
        df['label'] = df[column2].astype(str)
        if (df.ix[df['grand_total_sum'] < 0, :].shape[0]) != 1:
            df.ix[df['grand_total_sum'] < 0, :] = df.ix[df['grand_total_sum'] < 0, :].set_value(
                df.loc[df['grand_total_sum'] < 0, 'grand_total_sum'].index[0], 'grand_total_sum', 0.1)
        df.ix[df['new_total'] < 100, :] = df.ix[df['new_total'] < 100, :].set_value(
            df.loc[df['new_total'] < 100, 'new_total'].index, 'grand_total_sum', -0.1)
        df['invalid_ratio'] = df['invalid'] / df['new_total']
        df.loc[df['grand_total_sum'] < 0, 'label'] = 'others'
        if len(df.loc[df['grand_total_sum'] < 0, 'invalid_ratio']) != 0:
            df.loc[df['grand_total_sum'] < 0, 'invalid_ratio'] = ((df[df['grand_total_sum'] < 0]['invalid'].sum()) / (
                df[df['grand_total_sum'] < 0]['new_total'].sum()))
        df.sort_values('invalid_ratio', ascending=False, inplace=True)
        df['rank'] = df['invalid_ratio']
        df = df[[column1, column2, 'label', 'rank']]
        final = pd.concat([final, df])

    final.columns = [column1, column2, 'reason_xref_label', column1+"_"+column2+"_history"]
    final[column1] = final[column1].astype(str)
    final[column2] = final[column2].astype(str)

    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    train_Data_frame[column2] = train_Data_frame[column2].astype(str)
    train_Data_frame = pd.merge(train_Data_frame, final, how='left', on=[column1, column2])

    test_data_frame[column2] = test_data_frame[column2].astype(str)
    test_data_frame[column1] = test_data_frame[column1].astype(str)
    test_data_frame = pd.merge(test_data_frame, final, how='left', on=[column1, column2])

    del train_Data_frame[column1+"_"+column2+"_history"]
    del test_data_frame[column1+"_"+column2+"_history"]

    test_data_frame.loc[test_data_frame['reason_xref_label'].isnull(), 'reason_xref_label'] = 'others'

    train_Data_frame.loc[train_Data_frame['reason_xref_label'].isnull(), 'reason_xref_label'] = 'others'
    train_Data_frame = pd.merge(train_Data_frame, final.drop_duplicates([column1, 'reason_xref_label']),
                     on=[column1, 'reason_xref_label'], how='left')

    test_data_frame = pd.merge(test_data_frame, final.drop_duplicates([column1, 'reason_xref_label']),
                    on=[column1, 'reason_xref_label'], how='left')

    train_Data_frame.rename(columns={'REASON_CODE_x': column2}, inplace=True)
    test_data_frame.rename(columns={'REASON_CODE_x': column2}, inplace=True)


    return  train_Data_frame,test_data_frame

train,test=twoLevelHistory('fk_customer_map_id','ar_reason_code',train,test,80)



print(train['fk_customer_map_id_ar_reason_code_history'])
print(test['fk_customer_map_id_ar_reason_code_history'])
print(train['fk_customer_map_id_ar_reason_code_history'].isnull().sum())
print(test['fk_customer_map_id_ar_reason_code_history'].isnull().sum())
