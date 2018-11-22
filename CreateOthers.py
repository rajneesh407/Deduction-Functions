import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



def createTrainTest():
    #basicImports()
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

    train = data.loc[(data.deduction_created_date >= datetime.datetime(2015, 1, 2)) & (
            data.deduction_created_date <= datetime.datetime(2018, 5, 1))]
    # print(train.shape)
    # print(train)
    test = data.loc[data.deduction_created_date > datetime.datetime(2018, 1, 2)]

    return train,test


def getOthers(column1,train_Data_frame,test_Data_frame):
    #basicImports()

    train_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    train_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    test_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    test_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    test_Data_frame[column1] = test_Data_frame[column1].astype(str)
    grp_hstry_map_id = train_Data_frame.groupby(column1).agg({'output': 'sum', 'pk_deduction_id': 'count'})
    grp_hstry_map_id.reset_index(inplace=True)
    grp_hstry_map_id['output'] = grp_hstry_map_id['pk_deduction_id'] - grp_hstry_map_id['output']
    grp_hstry_map_id.columns = [column1, 'invalid', 'total']
    grp_hstry_map_id.sort_values(by='total', ascending=False, inplace=True)
    grp_hstry_map_id['cumsum'] = grp_hstry_map_id['total'].cumsum()
    grp_hstry_map_id['percentage'] = (grp_hstry_map_id['cumsum'] / (train_Data_frame.shape[0])) * 100
    grp_hstry_map_id.loc[grp_hstry_map_id['percentage'] > 90, column1] = "others"
    grp_list = grp_hstry_map_id[column1].unique().tolist()
    train_Data_frame[column1] = train_Data_frame[column1].apply(lambda x: x if (x in grp_list) else 'others')
    test_Data_frame[column1] = test_Data_frame[column1].apply(lambda x: x if (x in grp_list) else 'others')
    return train_Data_frame,test_Data_frame

train,test=createTrainTest()

print(train['payer'].nunique())

train,test=getOthers('payer',train,test)


print(train['payer'].nunique())
