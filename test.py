
def binaryMapping(binary_columns_list,df):
  list_binary=binary_columns_list
  for i in range(len(binary_columns_list)):
      print(df[list_binary[i]].dtype)
      print(list_binary[i])

      if(df[list_binary[i]].dtype=='object'):
          df[list_binary[i]] = df[list_binary[i]].apply(lambda x: int(x) if (pd.Series(x).dtype == 'float64' or pd.Series(x).dtype == 'int64') else x)
          temp_uniques=df[list_binary[i]].unique()

          df[list_binary[i]]=df[list_binary[i]].map({temp_uniques[0]:0,temp_uniques[1]:1})
      else:
          temp_uniques = df[list_binary[i]].unique()
          df[list_binary[i]] = df[list_binary[i]].map({temp_uniques[0]: 0, temp_uniques[1]: 1})
      print(df[list_binary[i]].value_counts(dropna=False))
  return  df 