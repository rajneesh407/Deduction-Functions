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





path=r"C:\Users\rajneesh.jha\Downloads\deduction\XGB\preliminary_model.xlsx"
def preliminaryModel(train,test,features,output_column,path):
    import numpy as np
    X_train = train[features]
    y_train = train[output_column]

    X_test = test[features]
    y_test = test[output_column]

    from xgboost.sklearn import XGBClassifier
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=200,
        max_depth=4,
        gamma=0,
        objective='binary:logistic',
        nthread=6,
        scale_pos_weight=0.6,
        seed=27)
    xgb1.fit(X_train, y_train)

    # pred_xgb1 = xgb1.predict(X_test)
    pred_proba_xgb1 = xgb1.predict_proba(X_test)[:, 1]
    pred_xgb1 = np.where(pred_proba_xgb1 > 0.80, 1, 0)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, pred_xgb1))
    import xgbfir

    xgbfir.saveXgbFI(xgb1, feature_names=features, OutputXlsxFile=path)


data=dataProcessing(data)
df=data
train,test=trainTestSplit(data)
features=['fk_customer_map_id_history',
 'fk_deduction_priority_id_history',
 'sold_to_party_history',
 'fk_reason_code_map_id_history',
 'fk_root_cause_code_id_history',
 'fk_deduction_owner_role_id_history',
 'fk_deduction_owner_user_id_history',
 'fk_processor_user_id_history',
 'fk_processor_role_id_history',
 'fk_document_type_map_id_history',
 'current_processor_deadline_date_history',
 'reminder_date_history',
 'customer_claim_date_history',
 'return_authorization_number_history',
 'pro_number_history',
 'reference_number3_history',
 'reference_number4_history',
 'reference_number5_history',
 'ref_date2_history',
 'deduction_closed_by_history',
 'deduction_created_by_history',
 'deduction_created_date_history',
 'deduction_change_by_history',
 'create_user_history',
 'update_user_history',
 'update_time_history',
 'vendor_name_history',
 'vendor_id_history',
 'commit_number_history',
 'payer_history',
 'posting_date_history',
 'discount_amount_history',
 'po_number_history',
 'order_number_history',
 'deal_id_history',
 'location_history',
 'due_date_history',
 'payment_term_history',
 'promortion_execution_from_date_history',
 'promotion_execution_to_date_history',
 'division_history',
 'job_id_history',
 'fiscal_year_history',
 'ar_reason_code_history',
 'customer_reason_code_history',
 'create_date_history',
 'update_date_history',
 'updated_columns_history',
 'check_date_history',
 'cal_credited_amount_history',
 'cal_write_off_amount_history',
 'cal_promotion_settlement_amount_history',
 'cal_reinstate_amount_history',
 'service_provider_history',
 'broker_id_history',
 'invoice_date_history',
 'auto_matched_commit_status_history',
 'fk_customer_map_id_partner1_history',
 'fk_customer_map_id_partner2_history',
 'buyer_name_history',
 'remittance_address_history',
 'reclamation_center_cost_norm_history',
 'control_number_history',
 'misccharges_norm_history',
 'handling_cost_norm_history',
 'customer_promotion_id_history',
 'payment_date_norm_history',
 'post_damage_cost_norm_history',
 'department_history',
 'days_out_standing_history',
 'cum_days_out_standing_history',
 'dos_offset_history',
 'days_since_worked_history',
 'fk_auto_matched_commit_status_history',
 'longname_history',
 'type_res_history',
 'res_amount_history',
 'header_amount_history',
 'item_amount_history',
 'resolution_type_history']

preliminaryModel(train,test,features,'output',path)