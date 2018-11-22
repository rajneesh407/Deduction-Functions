import pandas as pd
import numpy as np

data = pd.read_pickle(r"C:\Users\rajneesh.jha\Downloads\deduction\GP\three_tables_4_lakhs.pkl")

data = data.drop_duplicates('pk_deduction_id')
data.reset_index(inplace=True, drop=True)
df=data
def dataProcessing(data):
    # Replacing \N with np.nan`
    data = data.replace('\\N', np.nan)

    # Replacing None with np.nan
    data = data.fillna(value=np.nan)

    # Replacing the blank values
    data = data.replace(r'^\s*$', np.nan, regex=True)

    return data

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

print(data.shape)
data,list_of_null_features=is_Null_Mapping(data)
print(list_of_null_features)
data,list_of_binary_features=binaryMapping(list_of_binary())
print(list_of_binary_features)
print(data.shape)




















