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

