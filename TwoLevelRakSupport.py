import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

data = pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\three_tables_4_lakhs.pkl")

data = data.drop_duplicates('pk_deduction_id')
data.reset_index(inplace=True, drop=True)

data['fk_action_code_id'] = data['fk_action_code_id'].fillna(-1)
data.fk_action_code_id.value_counts(dropna=False)
data['fk_action_code_id'] = data['fk_action_code_id'].astype(int)
listid = [341, 419, 422, 425, 240, 243, 246, 428, 300]
data['output'] = np.nan
data.loc[(data.fk_action_code_id.isin(listid)), 'output'] = 0
data.output.fillna(value=1, inplace=True)
data['output'] = data.output.astype(int)

import datetime
data=data[data['fk_reason_category_id']==123]
alist=data[data['type_res'].isnull() & data['type_header'].isnull() & data['fk_action_code_id'].isnull()].pk_deduction_id.tolist()
data=data[~data['pk_deduction_id'].isin(alist)]
data.reset_index(inplace=True,drop=True)
train = data.loc[(data.deduction_created_date >= datetime.datetime(2015, 1, 2)) & (
            data.deduction_created_date <= datetime.datetime(2018, 3, 1))]
# print(train.shape)
# print(train)
test = data.loc[data.deduction_created_date >= datetime.datetime(2018, 3, 2)]

# print(train['fk_customer_map_id'].isnull().sum())
# print(train['sold_to_party'].isnull().sum())
#
# print(train['payer'].isnull().sum())


# print(test.shape)

def getOthers(column1,train_Data_frame,test_Data_frame,primary_id_column,output,value):
    #basicImports()

    train_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    train_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    test_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    test_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    test_Data_frame[column1] = test_Data_frame[column1].astype(str)
    grp_hstry_map_id = train_Data_frame.groupby(column1).agg({output: 'sum', primary_id_column: 'count'})
    grp_hstry_map_id.reset_index(inplace=True)
    grp_hstry_map_id[output] = grp_hstry_map_id[primary_id_column] - grp_hstry_map_id[output]
    grp_hstry_map_id.columns = [column1, 'invalid', 'total']
    grp_hstry_map_id.sort_values(by='total', ascending=False, inplace=True)
    grp_hstry_map_id['cumsum'] = grp_hstry_map_id['total'].cumsum()
    grp_hstry_map_id['percentage'] = (grp_hstry_map_id['cumsum'] / (train_Data_frame.shape[0])) * 100
    grp_hstry_map_id.loc[grp_hstry_map_id['percentage']>int(value), column1] = "others"
    grp_list = grp_hstry_map_id[column1].unique().tolist()
    train_Data_frame[column1] = train_Data_frame[column1].apply(lambda x: x if (x in grp_list) else 'others')
    test_Data_frame[column1] = test_Data_frame[column1].apply(lambda x: x if (x in grp_list) else 'others')
    return train_Data_frame,test_Data_frame

def handlePreprocessing(train_Data_frame, test_Data_frame, column1, column2):
    train_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    train_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    #train_Data_frame[column1] = train_Data_frame[column1].apply(
    #    lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    train_Data_frame[column1] = train_Data_frame[column1].astype(str)

    train_Data_frame[column2].replace('\\N', np.nan, inplace=True)
    train_Data_frame[column2].replace(np.nan, 'null', inplace=True)
    #train_Data_frame[column2] = train_Data_frame[column2].apply(
    #    lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    train_Data_frame[column2] = train_Data_frame[column2].astype(str)

    # test Data processing
    test_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    test_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    #test_Data_frame[column1] = test_Data_frame[column1].apply(
    #    lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_Data_frame[column1] = test_Data_frame[column1].astype(str)

    test_Data_frame[column2].replace('\\N', np.nan, inplace=True)
    test_Data_frame[column2].replace(np.nan, 'null', inplace=True)
    #test_Data_frame[column2] = test_Data_frame[column2].apply(
    #    lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_Data_frame[column2] = test_Data_frame[column2].astype(str)
    return train_Data_frame, test_Data_frame

def twoLevelRank(column1, column2, train_Data_frame, test_Data_frame,primary_key_column,output_column,value):
    # reason code and category level history
    train_Data_frame["temp0"] = train_Data_frame[column1]
    test_Data_frame["temp0"] = test_Data_frame[column1]

    train_Data_frame["temp1"]=train_Data_frame[column2]
    test_Data_frame["temp1"]=test_Data_frame[column2]
    train_Data_frame, test_Data_frame = getOthers(column1, train_Data_frame, test_Data_frame,primary_key_column,output_column,value)
    train_Data_frame, test_Data_frame = handlePreprocessing(train_Data_frame, test_Data_frame, column1, column2)

    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    train_Data_frame[column2] = train_Data_frame[column2].astype(str)

    reason_category = train_Data_frame.groupby([column1, column2]).agg(
        {output_column: 'sum', primary_key_column: 'count'}).sort_values(primary_key_column, ascending=False)

    reason_category.reset_index(inplace=True)

    reason_category.groupby([column1, column2]).agg({output_column: 'first', primary_key_column: 'first'})

    reason_category.reset_index(inplace=True)

    del reason_category['index']

    reason_category[output_column] = reason_category[primary_key_column] - reason_category[output_column]
    reason_category.columns = [column1, column2, 'invalid', 'new_total']

    rt_category = train_Data_frame.groupby([column1]).agg({primary_key_column: 'count'})
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
        df.ix[df['new_total'] < int(value), :] = df.ix[df['new_total'] < int(value), :].set_value(
            df.loc[df['new_total'] < int(value), 'new_total'].index, 'grand_total_sum', -0.1)
        df['invalid_ratio'] = df['invalid'] / df['new_total']
        df.loc[df['grand_total_sum'] < 0, 'label'] = 'others'
        if len(df.loc[df['grand_total_sum'] < 0, 'invalid_ratio']) != 0:
            df.loc[df['grand_total_sum'] < 0, 'invalid_ratio'] = ((df[df['grand_total_sum'] < 0]['invalid'].sum()) / (
                df[df['grand_total_sum'] < 0]['new_total'].sum()))
        df.sort_values('invalid_ratio', ascending=False, inplace=True)
        df['rank'] = df['invalid_ratio'].rank(ascending=False,method='dense')
        df = df[[column1, column2, 'label', 'rank']]
        final = pd.concat([final, df])

    #final.to_csv('table3.csv')
    final.columns = [column1, column2, column2+"_xref_label", column1 + "_" +column2 +"_rank"]
    final[column1] = final[column1].astype(str)
    final[column2] = final[column2].astype(str)

    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    train_Data_frame[column2] = train_Data_frame[column2].astype(str)
    train_Data_frame = pd.merge(train_Data_frame, final, how='left', on=[column1, column2])

    test_Data_frame[column2] = test_Data_frame[column2].astype(str)
    test_Data_frame[column1] = test_Data_frame[column1].astype(str)
    test_Data_frame = pd.merge(test_Data_frame, final, how='left', on=[column1, column2])

    del train_Data_frame[column1 + "_" +column2 +"_rank"]
    del test_Data_frame[column1 + "_" +column2 +"_rank"]

    test_Data_frame.loc[test_Data_frame[column2+"_xref_label"].isnull(), column2+"_xref_label"] = 'others'

    train_Data_frame.loc[train_Data_frame[column2+"_xref_label"].isnull(), column2+"_xref_label"] = 'others'
    train_Data_frame = pd.merge(train_Data_frame, final.drop_duplicates([column1, column2+"_xref_label"]),
                                on=[column1, column2+"_xref_label"], how='left')

    test_Data_frame = pd.merge(test_Data_frame, final.drop_duplicates([column1, column2+"_xref_label"]),
                               on=[column1, column2+"_xref_label"], how='left')

    train_Data_frame.rename(columns={column2+'_x': column2}, inplace=True)
    test_Data_frame.rename(columns={column2+'_x': column2}, inplace=True)
    del train_Data_frame[column1]
    del test_Data_frame[column1]

    train_Data_frame.rename(columns={'temp0':column1},inplace=True)
    test_Data_frame.rename(columns={'temp0': column1}, inplace=True)
    del train_Data_frame[column2+"_xref_label"]
    del test_Data_frame[column2+"_xref_label"]
    del train_Data_frame[column2+'_y']
    del test_Data_frame[column2+'_y']



    return train_Data_frame, test_Data_frame


def twoLevelRankSupport(two_lvel_list,train,test,primary_key_column,output,value):

    length=len(two_lvel_list)
    #print(length)
    for i in range(length):
        print(two_lvel_list[i][0],two_lvel_list[i][1])
        train,test=twoLevelRank(two_lvel_list[i][0],two_lvel_list[i][1],train,test,'pk_deduction_id','output','90')

    return train,test



x=[['payer','ar_reason_code'],['fk_customer_map_id','ar_reason_code'],['sold_to_party','fk_reason_code_map_id']]

train,test=twoLevelRankSupport(x,train,test,'pk_deduction_id','output',90)


print(train['fk_customer_map_id_ar_reason_code_rank'].isnull().sum())
print(test['fk_customer_map_id_ar_reason_code_rank'].isnull().sum())



print(train['sold_to_party_fk_reason_code_map_id_rank'].isnull().sum())
print(test['sold_to_party_fk_reason_code_map_id_rank'].isnull().sum())

print(train['payer_ar_reason_code_rank'].isnull().sum())
print(test['payer_ar_reason_code_rank'].isnull().sum())


