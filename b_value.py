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


def calculateBvalueRegularized(train, test, id_column, amount_column,primary_id_column,output_column,value):
    #from sklearn.cross_validation import StratifiedKFold
    from sklearn.model_selection import StratifiedKFold
    output_column_list = train[output_column].values
    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    stf.get_n_splits(train,output_column_list)
    train["temp0"] = train[id_column]
    test["temp0"] = test[id_column]

    train['b_value'] = np.nan
    train, test = getOthers(id_column, train, test,primary_id_column,output_column,value)
    for x_four_index, x_one_index in stf.split(train,output_column_list):
        x_four, x_one = train.iloc[x_four_index], train.iloc[x_one_index]
        print(x_four.shape, x_one.shape)
        # finding avg_ODA
        grp_avg_amt = x_four.groupby(id_column).agg({amount_column: 'mean'})
        grp_avg_amt.reset_index(inplace=True)
        grp_avg_amt.columns = [id_column, 'avg_oda']

        # finding avg_invalid_ODA
        grp_invalid_avg_amt = x_four[x_four[output_column] == 0].groupby(id_column).agg(
            {amount_column: 'mean'})
        grp_invalid_avg_amt.reset_index(inplace=True)
        grp_invalid_avg_amt.columns = [id_column, 'avg_invalid_oda']

        # classifying into low-high
        joined_valid_invalid = pd.merge(grp_avg_amt, grp_invalid_avg_amt, how='left', on=id_column)
        joined_valid_invalid.avg_invalid_oda.fillna(value=0, inplace=True)
        joined_valid_invalid['invalid_to_valid_ratio'] = joined_valid_invalid['avg_invalid_oda'] / joined_valid_invalid[
            'avg_oda']
        joined_valid_invalid['customer_level'] = np.where(joined_valid_invalid.invalid_to_valid_ratio > 1, 'high',
                                                          'low')

        # assigning b_val
        x_one = pd.merge(x_one, joined_valid_invalid, how='left', on=id_column)
        # x_one['b_value'] = x_one[amount_column ]/x_one['avg_invalid_oda']
        # x_one.loc[x_one.customer_level=='low', 'b_value']= 1/ x_one.loc[x_one.customer_level == 'low', 'b_value']

        x_one['b_value'] = np.where(x_one['avg_invalid_oda'] == 0.0, 0,
                                    x_one[amount_column] / x_one['avg_invalid_oda'])

        x_one['b_value'] = np.where(((x_one['customer_level'] == 'low') & (x_one['b_value'] != 0.0)),
                                    1 / x_one['b_value'],
                                    x_one['b_value'])

        train.loc[x_one_index, 'b_value'] = x_one['b_value'].values
        # train.loc[x_one_index,'avg_invalid_oda']=x_one['avg_invalid_oda'].values
        # train.loc[x_one_index,'customer_level']=x_one['customer_level'].values
    bval_df_common = pd.DataFrame.copy(train)
    # Calculate the avg oda
    bval_df_common[id_column] = bval_df_common[id_column].astype(str)
    grp_avg_amt_common = bval_df_common.groupby(id_column).agg({amount_column: 'mean'})
    grp_avg_amt_common.reset_index(inplace=True)
    grp_avg_amt_common.columns = [id_column, 'avg_oda']
    grp_avg_amt_common.tail()
    # Calculate the invalid mean ODA

    bval_df_common[id_column] = bval_df_common[id_column].astype(str)
    grp_inavlid_avg_amt_common = bval_df_common[bval_df_common[output_column] == 0].groupby(id_column).agg(
        {amount_column: 'mean'})
    grp_inavlid_avg_amt_common.reset_index(inplace=True)
    grp_inavlid_avg_amt_common.columns = [id_column, 'avg_invalid_oda']
    grp_inavlid_avg_amt_common.tail()
    # Joining them

    joined_valid_invalid_common = pd.merge(grp_avg_amt_common, grp_inavlid_avg_amt_common, how='left',
                                           on=id_column)
    joined_valid_invalid_common.avg_invalid_oda.fillna(value=0, inplace=True)
    joined_valid_invalid_common.tail()
    # Joining them

    joined_valid_invalid_common = pd.merge(grp_avg_amt_common, grp_inavlid_avg_amt_common, how='left',
                                           on=id_column)
    joined_valid_invalid_common.avg_invalid_oda.fillna(value=0, inplace=True)
    joined_valid_invalid_common.tail()

    # Classifying into high-low

    joined_valid_invalid_common['invalid_to_valid_ratio'] = joined_valid_invalid_common['avg_invalid_oda']/joined_valid_invalid_common['avg_oda']

    joined_valid_invalid_common['customer_level'] = np.where(joined_valid_invalid_common.invalid_to_valid_ratio > 1,
                                                             'high',
                                                             'low')
    # Joining with the test data
    joined_valid_invalid_common[id_column] = joined_valid_invalid_common[id_column].astype(str)
    test[id_column] = test[id_column].astype(str)

    test = pd.merge(test, joined_valid_invalid_common, on=id_column, how='left')
    test[amount_column] = test[amount_column].astype(float)
    # Calculating the b_value

    test['b_value'] = np.nan

    test['b_value'] = np.where(test['avg_invalid_oda'] == 0.0, 0,
                               test[amount_column] / test['avg_invalid_oda'])

    test['b_value'] = np.where(((test['customer_level'] == 'low') & (test['b_value'] != 0.0)), 1 / test['b_value'],
                               test['b_value'])

    del test['avg_invalid_oda']
    del test['customer_level']
    del test['avg_oda']
    del test['invalid_to_valid_ratio']
    #train.loc[train['b_value'].isnull(), 'b_value'] = 0.0
    #test.loc[test['b_value'].isnull(), 'b_value'] = 0.0
    del train[id_column]
    del test[id_column]

    train.rename(columns={'temp0':id_column},inplace=True)
    test.rename(columns={'temp0': id_column}, inplace=True)
    return train, test


def calculateBvalueUnregularized(train, test,id_column,amount_column,primary_id_column,output_column_name,value):
    train[id_column] = train.fk_customer_map_id.astype(str)
    test[id_column] = test.fk_customer_map_id.astype(str)
    train["temp0"] = train[id_column]
    test["temp0"] = test[id_column]

    train, test = getOthers(id_column, train, test, primary_id_column, output_column_name, value)

    grp_avg_amt = train.groupby(id_column).agg({amount_column: 'mean'})
    grp_avg_amt.reset_index(inplace=True)
    grp_avg_amt.columns = [id_column, 'avg_oda']

    # finding avg_invalid_ODA
    grp_invalid_avg_amt = train[train[output_column_name] == 0].groupby(id_column).agg(
        {amount_column: 'mean'})
    grp_invalid_avg_amt.reset_index(inplace=True)
    grp_invalid_avg_amt.columns = [id_column, 'avg_invalid_oda']

    joined_valid_invalid = pd.merge(grp_avg_amt, grp_invalid_avg_amt, how='left', on=id_column)
    joined_valid_invalid.avg_invalid_oda.fillna(value=0, inplace=True)
    joined_valid_invalid['invalid_to_valid_ratio'] = joined_valid_invalid['avg_invalid_oda'] / joined_valid_invalid['avg_oda']
    joined_valid_invalid['customer_level'] = np.where(joined_valid_invalid.invalid_to_valid_ratio > 1, 'high', 'low')

    # assigning b_val
    train = pd.merge(train, joined_valid_invalid, how='left', on=id_column)
    train['b_value'] = np.where(train['avg_invalid_oda'] == 0.0, 0,
                                train[amount_column] / train['avg_invalid_oda'])
    train['b_value'] = np.where(((train['customer_level'] == 'low') & (train['b_value'] != 0.0)), 1 / train['b_value'],
                                train['b_value'])
    test = pd.merge(test, joined_valid_invalid, on=id_column, how='left')

    test[amount_column] = test.original_dispute_amount.astype(float)

    # # Calculating the b_value

    test['b_value'] = 0.0
    test.loc[test['avg_invalid_oda'] == 0, 'b_value'] = 0.0
    test.loc[test['avg_invalid_oda'] != 0.0, 'b_value'] = test[amount_column] / test['avg_invalid_oda']
    test['b_value'] = np.where(((test['customer_level'] == 'low') & (test['b_value'] != 0.0)), 1 / test['b_value'],test['b_value'])

    train.loc[train['b_value'].isnull(),'b_value']=0.0
    test.loc[test['b_value'].isnull(),'b_value']=0.0
    del train[id_column]
    del test[id_column]

    train.rename(columns={'temp0': id_column}, inplace=True)
    test.rename(columns={'temp0': id_column}, inplace=True)

    return train, test


train,test=calculateBvalueRegularized(train,test,'fk_customer_map_id','pk_deduction_id','original_dispute_amount','output',90)


print(train['b_value'])
print(test['b_value'])
print(train.columns)
print(test.columns)
print(train['b_value'].isnull().sum())
print(test['b_value'].isnull().sum())
