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
   print("hello")
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
   print("r_value created")
   return train ,test
