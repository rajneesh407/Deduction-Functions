def createHistoryOneLevelLoo(column, data_frame, test_data_frame, primary_id_column, output, values):
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
    print(grp_column)

    # calculating history
    leave_one_out_sum = data_frame[column].map(data_frame.groupby(column)[output].sum())
    leave_one_out_count = data_frame[column].map(data_frame.groupby(column)[output].count())

    data_frame[column + "_history"] = ((leave_one_out_sum - data_frame[output])) / (leave_one_out_count - 1)
    data_frame[column + "_history"].fillna(0.0, inplace=True)
    # encoded_feature = train_new['mean_target'].values

    # merging with test
    grp_column = grp_column.groupby(column).agg({'invalid': 'sum', 'total': 'sum'}).reset_index()
    grp_column[column + "_history"] = grp_column['invalid'] / grp_column['total']

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

