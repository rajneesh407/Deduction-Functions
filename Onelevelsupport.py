
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

import datetime
data=data[data['fk_reason_category_id']==123]
alist=data[data['type_res'].isnull() & data['type_header'].isnull() & data['fk_action_code_id'].isnull()].pk_deduction_id.tolist()
data=data[~data['pk_deduction_id'].isin(alist)]
data.reset_index(inplace=True,drop=True)
train = data.loc[(data.deduction_created_date >= datetime.datetime(2015, 1, 2)) & (
            data.deduction_created_date <= datetime.datetime(2018, 3, 1))]
# print(train.shape)
# print(train)
test = data.loc[data.deduction_created_date > datetime.datetime(2018, 3, 2)]

print(train['fk_customer_map_id'].isnull().sum())
print(train['sold_to_party'].isnull().sum())

print(train['payer'].isnull().sum())




def createHistoryOneLevel(column, data_frame, test_data_frame,primary_id_column,output,values):
    grp_column = []
    data_frame['temp'] = data_frame[column]
    test_data_frame['temp_test'] = test_data_frame[column]

    data_frame[column].replace('\\N', np.nan, inplace=True)
    data_frame[column].replace(np.nan, 'null', inplace=True)
    data_frame[column] = data_frame[column].astype(str)
    grp_column = data_frame.groupby(column).agg({output: 'sum', primary_id_column: 'count'})
    grp_column.reset_index(inplace=True)
    grp_column[output] = grp_column[primary_id_column] - grp_column[output]
    grp_column.columns = [column, 'invalid', 'total']

    grp_column.sort_values(by='total', ascending=False, inplace=True)

    grp_column.sort_values(by='total', ascending=False, inplace=True)

    grp_column['cumsum'] = grp_column['total'].cumsum()
    grp_column['percentage'] = (grp_column['cumsum'] / (data_frame.shape[0])) * 100
    grp_column.loc[grp_column['percentage'] >= int(values), column] = 'others'
    grp_column[column] = grp_column[column].astype('str')

    grp_column_list = grp_column[column].unique().tolist()

    data_frame[column] = data_frame[column].apply(lambda x: x if (x in grp_column_list) else 'others')
    data_frame.reset_index(inplace=True)
    data_frame.drop('index', axis=1, inplace=True)

    # calculating history
    output_list = data_frame[output].values
    from sklearn.model_selection import StratifiedKFold
    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    stf.get_n_splits(data_frame, output_list)
    data_frame[str(column + "_history")] = np.nan

    for x_four_index, x_one_index in stf.split(data_frame, output_list):
        x_four, x_one = data_frame.iloc[x_four_index], data_frame.iloc[x_one_index]
        x_four['output1'] = x_four[output].map({1: 0, 0: 1})
        hstry_val = x_one[column].map(x_four.groupby(column).output1.mean())
        x_one['mean_hstry'] = hstry_val
        data_frame.loc[x_one_index, str(column + "_history")] = x_one['mean_hstry']

    # merging with test
    grp_column = grp_column.groupby(column).agg({'invalid': 'sum', 'total': 'sum'}).reset_index()
    grp_column[str(column + "_history")] = grp_column['invalid'] / grp_column['total']

    # test processing
    test_data_frame[column].replace('\\N', np.nan, inplace=True)
    test_data_frame[column].replace(np.nan, 'null', inplace=True)
    # test_data_frame[column] = test_data_frame[column].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_data_frame[column] = test_data_frame[column].astype(str)

    test_data_frame[column] = test_data_frame[column].apply(lambda x: x if (x in grp_column_list) else 'others')
    test_data_frame = pd.merge(test_data_frame, grp_column, on=column, how='left')

    del data_frame[column]
    del test_data_frame[column]

    data_frame.rename(columns={'temp': column}, inplace=True)
    test_data_frame.rename(columns={'temp_test': column}, inplace=True)
    print(column + " history created.")

    return data_frame, test_data_frame



def oneLevelSupport(one_level_list,train,test,primary_id_column,output,values):

    #lis=one_level_list
    features=[]
    for i in  one_level_list:
        train,test=createHistoryOneLevel(i,train,test,primary_id_column,output,values)
        print(train[i+"_history"].isnull().sum())
        print(test[i+"_history"].isnull().sum())
        if(train[i+"_history"].isnull().sum()+test[i+"_history"].isnull().sum())==0:
            features.append(str(i+"_history"))
    return train,test,features



x=['parent_deduction_id',
 'fk_customer_map_id',
 'fk_deduction_priority_id',
 'sold_to_party',
 'fk_reason_code_map_id',
 'fk_action_code_id',
 'fk_root_cause_code_id',
 'fk_deduction_owner_role_id',
 'fk_deduction_owner_user_id',
 'fk_processor_user_id',
 'fk_processor_role_id',
 'fk_document_type_map_id',
 'current_processor_deadline_date',
 'reminder_date',
 'ext_reference',
 'current_dispute_amount',
 'customer_claim_date',
 'return_authorization_number',
 'pro_number',
 'reference_number3',
 'reference_number4',
 'reference_number5',
 'reference_number7',
 'reference_number8',
 'ref_date2',
 'deduction_closed_by',
 'deduction_created_by',
 'deduction_created_date',
 'deduction_change_by',
 'create_user',
 'update_user',
 'update_time',
 'vendor_name',
 'vendor_id',
 'commit_number',
 'payer',
 'check_number',
 'posting_date',
 'discount_amount',
 'po_number',
 'order_number',
 'deal_id',
 'location',
 'due_date',
 'payment_term',
 'promortion_execution_from_date',
 'promotion_execution_to_date',
 'division',
 'job_id',
 'fiscal_year',
 'ar_reason_code',
 'customer_reason_code',
 'create_date',
 'update_date',
 'updated_columns',
 'check_date',
 'cal_credited_amount',
 'cal_write_off_amount',
 'cal_promotion_settlement_amount',
 'cal_reinstate_amount',
 'cal_unresolved_amount',
 'service_provider',
 'broker_id',
 'invoice_date',
 'auto_matched_commit_status',
 'fk_customer_map_id_partner1',
 'fk_customer_map_id_partner2',
 'buyer_name',
 'remittance_address',
 'reclamation_center_cost_norm',
 'control_number',
 'misccharges_norm',
 'handling_cost_norm',
 'customer_promotion_id',
 'payment_date_norm',
 'post_damage_cost_norm',
 'department',
 'days_out_standing',
 'cum_days_out_standing',
 'dos_offset',
 'days_since_worked',
 'last_user_action_date',
 'fk_auto_matched_commit_status',
 'longname',
 'type_res',
 'res_amount',
 'header_amount',
 'item_amount',
 'resolution_type']

#x=['fk_customer_map_id', 'payer', 'ar_reason_code', 'sold_to_party','fk_reason_code_map_id']
train,test,features=oneLevelSupport(x,train,test,'pk_deduction_id','output',92)

