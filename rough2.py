import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r"C:\Users\rajneesh.jha\Downloads\deduction\MARS\mars_data.csv")
data=data.drop_duplicates('Case ID')
data.reset_index(inplace=True,drop=True)
data.shape
# Replacing \N with np.nan`
data = data.replace('\\N', np.nan)

# Replacing None with np.nan
data = data.fillna(value=np.nan)

# Replacing the blank values
data = data.replace(r'^\s*$', np.nan, regex=True)
data['output']=1
data['REASON_CODE_aggregate']=data['REASON_CODE_aggregate'].astype(str)
data.loc[(data['Paid']>0)|( data['REASON_CODE_aggregate'].str.contains('10')),'output']=0
data['Ext_ref']=data['External refer.'].str[:2]
data['created_month']=pd.to_datetime(data['Created On']).dt.month
def history_train_test_timewise_split(data,column,history_start_date='2010-01-01',history_end_date='2050-12-31',train_start_date='2010-01-01',train_end_date='2050-12-31',test_start_date='2010-01-01',test_end_date='2050-12-31'):
    #created by: Manish
    #dependencies: pandas library
    #This method split data time wise and returns datasets for creating history or aggregate features, train dataset, left history i.e. part of dataset present in history but not in train,and test dataset .Dataframe, spliting criterion reference column, Start date and end date of respective splits  are required.
    #data- data frame
    #column= the column of the data frame to be used as reference column. It should not be null and contain dates.
    #history_start_date= start date for history. default='2010-01-01', value must be string and format should be 'yyyy-mm-dd'
    #history_end_date= end date for history. default='2050-01-01', value must be string and format should be 'yyyy-mm-dd'
    #train_start_date= start date for training dataset. default='2010-01-01', value must be string and format should be 'yyyy-mm-dd'
    #train_end_date= end date for training dataset. default='2050-01-01', value must be string and format should be 'yyyy-mm-dd'
    #test_start_date= start date for test dataset. default='2010-01-01', value must be string and format should be 'yyyy-mm-dd'
    #test_end_date= end date for test dataset. default='2050-01-01', value must be string and format should be 'yyyy-mm-dd'
    import pandas as pd
    try:
        history_start_date=pd.to_datetime(history_start_date)
        history_end_date=pd.to_datetime(history_end_date)
        train_start_date=pd.to_datetime(train_start_date)
        train_end_date=pd.to_datetime(train_end_date)
        test_start_date=pd.to_datetime(test_start_date)
        test_end_date=pd.to_datetime(test_end_date)
    except:
        raise ValueError('Either format is incorrect or date is in correct.Dates are expected as string and in "yyyy-mm-dd" format')
        return
    if data[column].isnull().sum()>0:
        raise ValueError('Split column contains null values')
        return
    else:
        data[column]=pd.to_datetime(data[column],dayfirst=True)
        history=data.loc[(data[column]>=history_start_date)&(data[column]<=history_end_date)]
        train=data.loc[(data[column]>=train_start_date)&(data[column]<=train_end_date)]
        test=data.loc[(data[column]>=test_start_date)&(data[column]<=test_end_date)]
        left_history=data.loc[(data[column]>=history_start_date)&(data[column]<train_start_date)]
        history.reset_index(drop=True,inplace=True)
        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)
        return history,train,left_history,test
train_start='2016-01-01'
train_end='2018-03-31'
history_start='2016-01-01'
history_end='2018-03-31'
test_start='2018-04-01'
train_range=train_start+"to"+train_end
test_range=test_start+"onwards"
history_range=history_start+"to"+history_end
dispute_history,train,left_history,test=history_train_test_timewise_split(data,
                                                                 'Created On',
                                                                 history_start_date=history_start,
                                                                 history_end_date=history_end,
                                                                 train_start_date=train_start,
                                                                 train_end_date=train_end,
                                                                 test_start_date=test_start)

train_241=train.loc[((train['Company Code']==241) & (train['Status']==40))]
test_241=test.loc[((test['Company Code']==241) & (test['Status']==40))]
test_241.loc[test_241['Original Disputed Amount']>0].shape
train=train_241
test=test_241


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



#------------------------------------------------------------------------------------------------------#

def dataProcessing(data):
    # Replacing \N with np.nan`
    data = data.replace('\\N', np.nan)

    # Replacing None with np.nan
    data = data.fillna(value=np.nan)

    # Replacing the blank values
    data = data.replace(r'^\s*$', np.nan, regex=True)

    return data


def list_of_binary():

    summary=get_summary(data.columns)
    #features with two uniques
    summary_binary=summary[(summary.no_of_uniques==2 ) & (summary['null_percentage']<('0.01 %'))]
    list_of_binary_features = summary_binary.column_name.tolist()
    return list_of_binary_features


#map values to 1 and 0
def binaryMapping(binary_columns_list):
  list_binary=binary_columns_list
  list_of_binary_features=[]
  for i in range(len(binary_columns_list)):
      # print(df[list_binary[i]].dtype)
      # print(list_binary[i])

      if(data[list_binary[i]].dtype=='object'):
          data[list_binary[i]] = data[list_binary[i]].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64' or pd.Series(x).dtype == 'int64') else x)
          temp_uniques=data[list_binary[i]].unique()

          data[list_binary[i]]=data[list_binary[i]].map({temp_uniques[0]:0,temp_uniques[1]:1})
      else:
          temp_uniques = data[list_binary[i]].unique()
          data[list_binary[i]] = data[list_binary[i]].map({temp_uniques[0]: 0, temp_uniques[1]: 1})
      list_of_binary_features.append(list_binary[i])
  return data,list_of_binary_features


def createNulls(column1,data):
    #data[column1]=data[column1].astype('str')
    #print(data[column1])
    data[column1+"_isnull"]=np.nan

    data[column1+"_isnull"]= data[column1].map({np.nan:0})
    data[column1+"_isnull"].fillna(1,inplace=True)
    return data



def is_Null_Mapping(data):
    summary=get_summary(data.columns)
    columns=summary['column_name'].tolist()
    length=len(columns)
    summary['null_percentage']=summary['null_percentage'].astype(str)
    list_of_isnull_features=[]
    for i in range(length):
        # x=summary.loc[summary['column_name']==columns[i]]
        # print(x.columns.tolist())
        # x['null_percentage']=x['null_percentage'].astype('str')
        # print(x['null_percentage'].dtype)
        # if (x[x['column_name']==columns[i]]['null_percentage']!='0.00%' & x[x['column_name']==columns[i]]['null_percentage']!='100.00%' ):
        #    print(columns[i])

        if( (data[columns[i]].isnull().sum() > 0) & (data[columns[i]].isnull().sum()!=data.shape[0])):
            data=createNulls(columns[i],data)
            list_of_isnull_features.append(columns[i]+"_isnull")
    return  data,list_of_isnull_features


def startOneLevelHistoryList(data,train,test):
    summary=get_summary(data.columns.tolist())

    x=summary.loc[(summary['no_of_uniques']>3 )& (summary['no_of_uniques']<(data.shape[0]/4) )]['column_name'].tolist()
    return x




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




def oneLevelSupport(one_level_list,remove_column_list,train,test,primary_id_column,output,values):

    #lis=one_level_list
    features=[]
    for i in  one_level_list:
        if i in remove_column_list:
            pass
        else :
            train,test=createHistoryOneLevel(i,train,test,primary_id_column,output,values)
            print(train[i+"_history"].isnull().sum())
            print(test[i+"_history"].isnull().sum())
            if(train[i+"_history"].isnull().sum()+test[i+"_history"].isnull().sum())==0:
                features.append(str(i+"_history"))
                prelim_features.append(str(i+"_history"))
    return train,test,features


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

import xlrd
import operator

def individualInteraction(first_sheet):
      df={}
      for i in range(7):
          df_temp = {}
          for row in range(1,first_sheet.nrows):
              df_temp[first_sheet.row_values(row)[0]]=first_sheet.row_values(row)[7+i]
          df_temp_soted=sorted(df_temp.items(),  key=lambda x: x[1])

          temp_list=[i[0] for i in df_temp_soted]
          df[first_sheet.row_values(0)[7+i]]=temp_list
      return df




def  oneLevelHistory(second_sheet):
     df = {}
     for i in range(7):
         df_temp = {}
         for row in range(1, second_sheet.nrows):
             if (second_sheet.row_values(row)[0].split("|")[0]) != (second_sheet.row_values(row)[0].split("|")[1]):
              df_temp[second_sheet.row_values(row)[0]] = second_sheet.row_values(row)[7 + i]
         df_temp_soted = sorted(df_temp.items(), key=lambda x: x[1])

         temp_list = [i[0] for i in df_temp_soted]
         #print(temp_list)
         df[second_sheet.row_values(0)[7 + i]] = temp_list

     return df


def twoLevelHIstory(third_sheet):
    df = {}
    for i in range(7):
        df_temp = {}
        for row in range(1, third_sheet.nrows):
            if ((third_sheet.row_values(row)[0].split("|")[0]) != (third_sheet.row_values(row)[0].split("|")[1])
                and (third_sheet.row_values(row)[0].split("|")[1]) != (third_sheet.row_values(row)[0].split("|")[2])
                and (third_sheet.row_values(row)[0].split("|")[2]) != (third_sheet.row_values(row)[0].split("|")[0])
                ):
                df_temp[third_sheet.row_values(row)[0]] = third_sheet.row_values(row)[7 + i]
        df_temp_soted = sorted(df_temp.items(), key=lambda x: x[1])

        temp_list = [i[0] for i in df_temp_soted]
        df[third_sheet.row_values(0)[7 + i]] = temp_list
    return df


def interactions(path,individual_rank,onelevel_rank,twolevel_rank):

    workbook=xlrd.open_workbook(path)
    indivual_interaction=individualInteraction(workbook.sheet_by_index(0))
    one_level_interaction=oneLevelHistory(workbook.sheet_by_index(1))
    two_level_interaction=twoLevelHIstory(workbook.sheet_by_index(2))

    length_one=len(one_level_interaction[onelevel_rank])
    one_level=[]
    for i in range(length_one):
        x=[]
        x.append(one_level_interaction[onelevel_rank][i].split("|")[0])
        x.append(one_level_interaction[onelevel_rank][i].split("|")[1])
        one_level.append(x)

    two_level = []
    length_two=len(two_level_interaction[twolevel_rank])
    for i in range(length_two):
        x = []
        x.append(two_level_interaction[twolevel_rank][i].split("|")[0])
        x.append(two_level_interaction[twolevel_rank][i].split("|")[1])
        x.append(two_level_interaction[twolevel_rank][i].split("|")[1])

        two_level.append(x)

    #print(one_level)
    #print(two_level)
    return indivual_interaction[individual_rank],one_level,two_level
def column_converter(li):
    string = '_history'
    for i in range(len(li)):
        for j in range(len(li[i])):
            if string in li[i][j]:
                li[i][j] = li[i][j][:li[i][j].index('_history')]
    return li


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
#--------------------------------------------------------------------------------------------#
def twoLevelRank(column1, column2, train_Data_frame, test_Data_frame,primary_key_column,output_column,value):
    # reason code and category level history
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
    #train_Data_frame.rename(columns={'temp1':column2},inplace=True)
    #test_Data_frame.rename(columns={'temp1': column2}, inplace=True)
    del train_Data_frame[column2+"_xref_label"]
    del test_Data_frame[column2+"_xref_label"]
    del train_Data_frame[column2+'_y']
    del test_Data_frame[column2+'_y']



    return train_Data_frame, test_Data_frame


def twoLevelRankSupport(two_lvel_list,train,test,primary_key_column,output,value):

    length=len(two_lvel_list)
    #print(length)
    for i in range(length):
        #print(two_lvel_list[i][0],two_lvel_list[i][1])
        train,test=twoLevelRank(two_lvel_list[i][0],two_lvel_list[i][1],train,test,primary_key_column,output,value)
        #print(test['ar_reason_code'].value_counts(dropna=False))
        prelim_features.append(two_lvel_list[i][0] + "_" + two_lvel_list[i][1] + "_rank")
        print(two_lvel_list[i][0] + "_" + two_lvel_list[i][1] + "_rank  created")
    return train,test


#-------------------------------------------------------------------------------------------------------#
def twoLevelHistory(column1, column2, train_Data_frame, test_Data_frame,primary_key_column,output_column,value):
    # reason code and category level history
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
        df['rank'] = df['invalid_ratio']
        df = df[[column1, column2, 'label', 'rank']]
        final = pd.concat([final, df])

    #final.to_csv('table3.csv')
    final.columns = [column1, column2, column2+"_xref_label", column1 + "_" + column2 + "_history"]
    final[column1] = final[column1].astype(str)
    final[column2] = final[column2].astype(str)

    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    train_Data_frame[column2] = train_Data_frame[column2].astype(str)
    train_Data_frame = pd.merge(train_Data_frame, final, how='left', on=[column1, column2])

    test_Data_frame[column2] = test_Data_frame[column2].astype(str)
    test_Data_frame[column1] = test_Data_frame[column1].astype(str)
    test_Data_frame = pd.merge(test_Data_frame, final, how='left', on=[column1, column2])

    del train_Data_frame[column1 + "_" + column2 + "_history"]
    del test_Data_frame[column1 + "_" + column2 + "_history"]

    test_Data_frame.loc[test_Data_frame[column2+"_xref_label"].isnull(), column2+"_xref_label"] = 'others'

    train_Data_frame.loc[train_Data_frame[column2+"_xref_label"].isnull(), column2+"_xref_label"] = 'others'
    train_Data_frame = pd.merge(train_Data_frame, final.drop_duplicates([column1, column2+"_xref_label"]),
                                on=[column1, column2+"_xref_label"], how='left')

    test_Data_frame = pd.merge(test_Data_frame, final.drop_duplicates([column1, column2+"_xref_label"]),
                               on=[column1, column2+"_xref_label"], how='left')


    train_Data_frame.rename(columns={column2+'_x': column2}, inplace=True)
    test_Data_frame.rename(columns={column2+'_x': column2}, inplace=True)
    #train_Data_frame.rename(columns={'temp1':column2},inplace=True)
    #test_Data_frame.rename(columns={'temp1': column2}, inplace=True)
    del train_Data_frame[column2+"_xref_label"]
    del test_Data_frame[column2+"_xref_label"]
    del train_Data_frame[column2+'_y']
    del test_Data_frame[column2+'_y']

    print(column1+" "+column2+" history created")

    return train_Data_frame, test_Data_frame






def twoLevelHistorySupport(two_lvel_list,train,test,primary_key_column,output,value):

    length=len(two_lvel_list)
    features_two_level=[]
    for i in range(length):
        train,test=twoLevelHistory(two_lvel_list[i][0],two_lvel_list[i][1],train,test,primary_key_column,'output','90')
        features_two_level.append(two_lvel_list[i][0]+"_"+two_lvel_list[i][1]+"_history")
        prelim_features.append(two_lvel_list[i][0] + "_" + two_lvel_list[i][1] + "_history")
        #print(two_lvel_list[i][0] + "_" + two_lvel_list[i][1] + "_history  created")
    return train,test,features_two_level

#------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------#

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
    prelim_features.append('b_value')
    print("b_value created")
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
    prelim_features.append('b_value')

    return train, test

#-------------------------------------------------------------------------------------------------#

def rowLevelHistory(column_name, primary_key_name, amount_column, output_column, train, test,value):
    train[column_name] = train[column_name].astype(str)
    train["temp0"] = train[column_name]
    test["temp0"] = test[column_name]

    train, test = getOthers(column_name, train, test,primary_key_name,output_column,value)
    avg_amt_df = train.groupby(column_name).agg({amount_column: 'mean'}).reset_index()
    avg_amt_df.columns = [column_name, 'mean_oda']
    avg_amt_df.tail()
    train = pd.merge(train, avg_amt_df, on=column_name, how='left')
    train['amount_label'] = np.where(train[amount_column] >= train['mean_oda'], 'high', 'low')

    output_list = train[output_column].values
    train['row_historyy'] = np.nan
    from sklearn.model_selection import StratifiedKFold

    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    stf.get_n_splits(train, output_list)

    for x_four_index, x_one_index in stf.split(train, output_list):
        x_four, x_one = train.iloc[x_four_index], train.iloc[x_one_index]

        temp = x_four.groupby([column_name, 'amount_label']).agg(
            {output_column: 'sum', primary_key_name: 'count'}).reset_index()
        temp[output_column] = temp[primary_key_name] - temp[output_column]
        temp.rename(columns={output_column: 'invalid', primary_key_name: 'total'}, inplace=True)
        temp['row_level_history'] = temp['invalid'] / temp['total']
        temp.drop(['invalid', 'total'], axis=1, inplace=True)
        x_one = pd.merge(x_one, temp, on=[column_name, 'amount_label'], how='left')
        x_one.set_index(x_one_index, inplace=True)
        train.loc[x_one_index, 'row_history'] = x_one['row_level_history']
    row_level_df = pd.DataFrame.copy(train)
    row_level_df[column_name] = row_level_df[column_name].astype(str)
    avg_amt_df_tr = row_level_df.groupby(column_name).agg({amount_column: 'mean'}).reset_index()
    avg_amt_df_tr.columns = [column_name, 'mean_oda']
    avg_amt_df_tr.tail()
    row_level_df = pd.merge(row_level_df, avg_amt_df_tr, on=column_name, how='left')
    row_level_df.shape
    row_level_df_grp = row_level_df.groupby([column_name, 'amount_label']).agg(
        {output_column: 'sum', primary_key_name: 'count'})
    row_level_df_grp.reset_index(inplace=True)
    row_level_df_grp[output_column] = row_level_df_grp[primary_key_name] - row_level_df_grp[output_column]
    row_level_df_grp.columns = [column_name, 'amount_label', 'invalid', 'total']
    row_level_df_grp['row_history'] = row_level_df_grp['invalid'] / row_level_df_grp['total']
    row_level_df_grp.drop(['invalid', 'total'], axis=1, inplace=True)
    row_level_df_grp.tail()

    test[column_name] = test[column_name].astype(str)
    avg_amt_df_tst = test.groupby(column_name).agg({amount_column: 'mean'}).reset_index()
    avg_amt_df_tst.columns = [column_name, 'mean_oda']
    avg_amt_df_tst.tail()
    test = pd.merge(test, avg_amt_df_tst, on=column_name, how='left')
    test.shape
    # Classifying them as low and high

    test['amount_label'] = np.where(test[amount_column] >= test['mean_oda'], 'high', 'low')
    # joining with the test data

    test = pd.merge(test, row_level_df_grp, on=[column_name, 'amount_label'], how='left')
    del train[column_name]
    del test[column_name]

    train.rename(columns={'temp0': column_name}, inplace=True)
    test.rename(columns={'temp0': column_name}, inplace=True)
    prelim_features.append('row_history')
    print("row level history created")
    return train, test




#-----------------------------------------------------------------------------------------------#
def createHistoryOneLevelMeanExpanding(column, data_frame, test_data_frame, primary_id_column, output, values):
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
    print(grp_column[column].dtype)
    grp_column_list = grp_column[column].unique().tolist()

    data_frame[column] = data_frame[column].apply(lambda x: x if (x in grp_column_list) else 'others')
    data_frame.reset_index(inplace=True)
    data_frame.drop('index', axis=1, inplace=True)
    print(grp_column)

    # calculating history
    data_frame[str(column + "_history")] = np.nan
    cumsum = data_frame.groupby(column)[output].cumsum() - data_frame[output]
    cumcnt = data_frame.groupby(column).cumcount()
    print(cumsum, cumcnt)
    data_frame[column + "_history"] = cumsum / cumcnt
    data_frame[column + "_history"].fillna(0.0, inplace=True)

    # encoded_feature = train_new['mean_target'].values

    # merging with test
    grp_column = grp_column.groupby(column).agg({'invalid': 'sum', 'total': 'sum'}).reset_index()
    grp_column[str(column + "_history")] = grp_column['invalid'] / grp_column['total']

    # test processing
    test_data_frame[column].replace('\\N', np.nan, inplace=True)
    test_data_frame[column].replace(np.nan, 'null', inplace=True)
    # test_data_frame[column] = test_data_frame[column].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_data_frame[column] = test_data_frame[column].astype(str)
    # print(test_data_frame[column].nunique())
    grp_column_list = [str(i) for i in grp_column_list]

    test_data_frame[column] = test_data_frame[column].apply(lambda x: x if (x in grp_column_list) else 'others')
    test_data_frame = pd.merge(test_data_frame, grp_column, on=column, how='left')

    del data_frame[column]
    del test_data_frame[column]

    data_frame.rename(columns={'temp': column}, inplace=True)
    test_data_frame.rename(columns={'temp_test': column}, inplace=True)
    print(column + " history created.")
    return data_frame, test_data_frame



def oneLevelSupportExpanding(one_level_list,train,test,primary_id_column,output,values):

    #lis=one_level_list
    features=[]
    for i in  one_level_list:
        train,test=createHistoryOneLevelMeanExpanding(i,train,test,primary_id_column,output,values)
        print(train[i+"_history"].isnull().sum())
        print(test[i+"_history"].isnull().sum())
        if(train[i+"_history"].isnull().sum()+test[i+"_history"].isnull().sum())==0:
            prelim_features.append(str(i+"_history"))
    return train,test,prelim_features

def r_value(column_name, train, test, amount_column,  primary_column, output, value):
   train["temp0"] = train[column_name]
   test["temp0"] = test[column_name]
   train[column_name]=train[column_name].astype(str)
   test[column_name]=test[column_name].astype(str)
   train, test = getOthers(column_name, train, test, primary_column, output, value)
   amnt = train[train[output] == 0].groupby(column_name).agg({amount_column: 'sum'})
   amnt.columns = ['amount_invalid_sum']
   amnt.reset_index(inplace=True)
   x = train.groupby([column_name]).agg({primary_column: 'count', output: 'sum'})
   x['invalid'] = x[primary_column] - x[output]
   x.columns = ['total', 'valid', 'invalid']
   x.reset_index(inplace=True)
   combined_frame = pd.merge(x, amnt, on=column_name, how='left')
   combined_frame['amount_invalid_sum'].fillna(0.0, inplace=True)
   total_data_invalid_amount = train[amount_column].sum()
   total_data_invalid_count = train[train[output] == 0][output].count()
   combined_frame['r_value'] = ((combined_frame['amount_invalid_sum'] / (total_data_invalid_amount)) / (
               combined_frame['invalid'] / (total_data_invalid_count)))
   combined_frame['r_value'].fillna(0.0, inplace=True)
   del combined_frame['total']
   del combined_frame['valid']
   del combined_frame['invalid']
   del combined_frame['amount_invalid_sum']

   train = pd.merge(train, combined_frame, on=column_name, how='left')
   test = pd.merge(test, combined_frame, on=column_name, how='left')
   test['r_value'].fillna(0.0, inplace=True)
   train['r_value'].fillna(0.0, inplace=True)
   del train[column_name]
   del test[column_name]

   train.rename(columns={'temp0': column_name}, inplace=True)
   test.rename(columns={'temp0': column_name}, inplace=True)
   prelim_features.append("r_value")
   print("r_value created")
   return train ,test


#--------------------------------------------------------------------------------------#












model_dict={}
model_dict['Customer']=['Customer']
model_dict['primary_key_column']=['Case ID']
model_dict['output']=['output']
model_dict['dates']=['deduction_created_date']
model_dict['entity']=['Ext_ref','Customer','Reason','Category']
model_dict['amount']=['Original Disputed Amount']




def getDictList(model_dict,key_name):
    return model_dict[key_name]

path=r"C:\Users\rajneesh.jha\Downloads\deduction\XGB\preliminary_model.xlsx"
prelim_features=[]
df = data

def processData(data):
    data=dataProcessing(data)
    data,list_of_null_features=is_Null_Mapping(data)
    print("null features created")
    print(list_of_null_features)
    return data
    #data,list_of_binary_features=binaryMapping(list_of_binary())
#one_level_list=startOneLevelHistoryList(data,train,test)
def xgbmodelling(model_dict,train,test,value):
    one_level_list=getDictList(model_dict,'entity')
    remove_column_list=[]

    primary_key_column=getDictList(model_dict,'primary_key_column')
    customer_column=getDictList(model_dict,'Customer')
    amount_column=getDictList(model_dict,'amount')
    output=getDictList(model_dict,'output')
    primary_key_column=str(primary_key_column[0])
    customer_column=str(customer_column[0])
    output=str(output[0])
    amount_column=str(amount_column[0])

    train,test,features_one_level=oneLevelSupport(one_level_list,remove_column_list,train,test,primary_key_column,output,value)
    preliminaryModel(train,test,features_one_level,output,path)
    one_level_feature_list,two,three=interactions(path,'Average Rank','Gain Rank','FScore Rank')
    two_level_feature_list=column_converter(two)
    three_level_feature_list=column_converter(three)
    train,test=twoLevelRankSupport(two_level_feature_list,train,test,primary_key_column,output,value)
    train,test,features_two_level=twoLevelHistorySupport(two_level_feature_list,train,test,primary_key_column,output,value)
    train,test=calculateBvalueRegularized(train,test,customer_column,amount_column,primary_key_column,output,value)
    train,test=rowLevelHistory(customer_column,primary_key_column,amount_column,output,train,test,value)
    train,test=r_value(customer_column,train,test,amount_column,primary_key_column,output,value)
    return train,test


def featureEngineering(data,train,test,value):

    data=processData(data)
    train,test=xgbmodelling(model_dict,train,test,value)
    return train,test

train,test=featureEngineering(data,train,test,90)
print(prelim_features)

