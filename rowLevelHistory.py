def getOthers(column1, train_Data_frame, test_Data_frame,primary_id_column,output_column,value):
    train_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    train_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    train_Data_frame[column1] = train_Data_frame[column1].apply(
        lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    print(train_Data_frame[column1].value_counts(dropna=False))
    print(test_Data_frame[column1].value_counts(dropna=False))

    # test Data processing
    test_Data_frame[column1].replace('\\N', np.nan, inplace=True)
    test_Data_frame[column1].replace(np.nan, 'null', inplace=True)
    test_Data_frame[column1] = test_Data_frame[column1].apply(
        lambda x: int(x) if (pd.Series(x).dtype == 'float64') else x)
    test_Data_frame[column1] = test_Data_frame[column1].astype(str)

    train_Data_frame[column1] = train_Data_frame[column1].astype(str)
    grp_hstry_map_id = train_Data_frame.groupby(column1).agg({output_column: 'sum', primary_id_column: 'count'})
    grp_hstry_map_id.reset_index(inplace=True)
    grp_hstry_map_id[output_column] = grp_hstry_map_id[primary_id_column] - grp_hstry_map_id[output_column]
    grp_hstry_map_id.columns = [column1, 'invalid', 'total']
    grp_hstry_map_id.sort_values(by='total', ascending=False, inplace=True)
    grp_hstry_map_id.loc[grp_hstry_map_id.total <= int(value), column1] = 'others'
    grp_hstry_map_id[column1] = grp_hstry_map_id[column1].astype(str)

    grp_cust_list = grp_hstry_map_id[column1].unique().tolist()
    len(grp_cust_list)
    train_Data_frame[column1] = train[column1].apply(lambda x: x if (x in grp_cust_list) else 'others')
    train_Data_frame.reset_index(inplace=True)
    train_Data_frame.drop('index', axis=1, inplace=True)

    test_Data_frame[column1] = test_Data_frame[column1].apply(lambda x: x if (x in grp_cust_list) else 'others')
    return train_Data_frame, test_Data_frame

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
        print(x_four.shape, x_one.shape)

        temp = x_four.groupby([column_name, 'amount_label']).agg(
            {output_column: 'sum', primary_key_name: 'count'}).reset_index()
        temp[output_column] = temp[primary_key_name] - temp[output_column]
        temp.rename(columns={output_column: 'invalid', primary_key_name: 'total'}, inplace=True)
        temp['row_level_history'] = temp['invalid'] / temp['total']
        temp.drop(['invalid', 'total'], axis=1, inplace=True)
        print(temp.shape)
        x_one = pd.merge(x_one, temp, on=[column_name, 'amount_label'], how='left')
        x_one.set_index(x_one_index, inplace=True)
        print(x_one.shape)
        # x_one.reset_index(inplace=True)
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
    return train, test


