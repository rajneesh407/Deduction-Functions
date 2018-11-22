import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import model_selection

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
def get_history(x):
    return (1-x.mean())

def get_invalid_counts(x):
    return (len(x)-x.sum())

def get_group_column(z,top):
    return z.apply(lambda x: x if (x in top) else 'others')

def dispute_amount_deviation(history, left_history, train, test, customer_identifier, dispute_amount,
                             output_label='output_label', fill_null='others', customer_grouping_threshold=500,
                             regularized=False):
    # preprocessing of columns
    left_history[customer_identifier].fillna(fill_null, inplace=True)
    train[customer_identifier].fillna(fill_null, inplace=True)
    test[customer_identifier].fillna(fill_null, inplace=True)
    history[customer_identifier].fillna(fill_null, inplace=True)
    history[customer_identifier] = history[customer_identifier].astype(str)
    train[customer_identifier] = train[customer_identifier].astype(str)
    test[customer_identifier] = test[customer_identifier].astype(str)
    left_history[customer_identifier] = left_history[customer_identifier].astype(str)

    # grouping top customer for history data
    all_values = history[customer_identifier].value_counts()
    top_values = all_values[all_values >= customer_grouping_threshold].index.tolist()
    print(top_values)
    history[customer_identifier] = get_group_column(history[customer_identifier], top_values)

    train[customer_identifier] = get_group_column(train[customer_identifier], top_values)
    test[customer_identifier] = get_group_column(test[customer_identifier], top_values)

    left_history[customer_identifier] = get_group_column(left_history[customer_identifier], top_values)
    # calculating average dispute amount per customer

    average_dispute_amount_per_customer = history.groupby(customer_identifier).agg({dispute_amount: 'mean'})
    average_dispute_amount_per_customer.reset_index(inplace=True)
    average_dispute_amount_per_customer.columns = [customer_identifier, 'average_dispute_amount']
    test = pd.merge(test, average_dispute_amount_per_customer, on=customer_identifier, how='left')

    test['original_by_average_dispute_amount'] = test[dispute_amount] / test['average_dispute_amount']

    if regularized == True:

        y_train = train[output_label]
        skf = model_selection.StratifiedKFold(5, shuffle=True, random_state=0)
        print(y_train.isnull().sum())
        for x_four_index, x_one_index in skf.split(train, y_train):
            train_fold, val_fold = train.iloc[x_four_index], train.iloc[x_one_index]
            train_fold = pd.concat([train_fold, left_history], axis=0)
            # finding avg_ODA
            average_dispute_amount_per_customer = train_fold.groupby(customer_identifier).agg({dispute_amount: 'mean'})
            average_dispute_amount_per_customer.reset_index(inplace=True)
            average_dispute_amount_per_customer.columns = [customer_identifier, 'average_dispute_amount_per_customer']
            val_fold = pd.merge(val_fold, average_dispute_amount_per_customer, on=customer_identifier, how='left')
            val_fold['average_dispute_amount_per_customer'].fillna(val_fold[dispute_amount], inplace=True)
            val_fold['original_by_average_dispute_amount'] = val_fold[dispute_amount] / val_fold[
                'average_dispute_amount_per_customer']
            train.loc[x_one_index, 'original_by_average_dispute_amount'] = val_fold[
                'original_by_average_dispute_amount'].values
    else:
        train = pd.merge(train, average_dispute_amount_per_customer, on=customer_identifier, how='left')
        train['original_by_average_dispute_amount'] = train[dispute_amount] / train['average_dispute_amount']

    train['original_by_average_dispute_amount'].fillna(1, inplace=True)
    test['original_by_average_dispute_amount'].fillna(1, inplace=True)
    train['original_by_average_dispute_amount'] = train['original_by_average_dispute_amount'].astype(np.float64)
    test['original_by_average_dispute_amount'] = test['original_by_average_dispute_amount'].astype(np.float64)
    return train, test, average_dispute_amount_per_customer





train,test,a=dispute_amount_deviation(train,train,train,test,'fk_customer_map_id','original_dispute_amount','output','others',100,True)
print(train['original_by_average_dispute_amount'],test['original_by_average_dispute_amount'])
print(train['original_by_average_dispute_amount'].isnull().sum(),test['original_by_average_dispute_amount'].isnull().sum())
