def calculateBvalueRegularized(train, test, id_column, amount_column,primary_id_column,output_column,value):
    #from sklearn.cross_validation import StratifiedKFold
    from sklearn.model_selection import StratifiedKFold
    output_column_list = train[output_column].values
    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    stf.get_n_splits(train,output_column_list)
    train["temp0"] = train[id_column]
    test["temp0"] = test[id_column]

    train['reason_b_value'] = np.nan
    train, test = getOthers(id_column, train, test,primary_id_column,output_column,value)
    for x_four_index, x_one_index in stf.split(train,output_column_list):
        x_four, x_one = train.iloc[x_four_index], train.iloc[x_one_index]
        print(x_four.shape, x_one.shape)
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
        # x_one['reason_b_value'] = x_one[amount_column ]/x_one['avg_invalid_oda']
        # x_one.loc[x_one.customer_level=='low', 'reason_b_value']= 1/ x_one.loc[x_one.customer_level == 'low', 'reason_b_value']

        x_one['reason_b_value'] = np.where(x_one['avg_invalid_oda'] == 0.0, 0,
                                    x_one[amount_column] / x_one['avg_invalid_oda'])

        x_one['reason_b_value'] = np.where(((x_one['customer_level'] == 'low') & (x_one['reason_b_value'] != 0.0)),
                                    1 / x_one['reason_b_value'],
                                    x_one['reason_b_value'])

        train.loc[x_one_index, 'reason_b_value'] = x_one['reason_b_value'].values
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

    test['reason_b_value'] = np.nan

    test['reason_b_value'] = np.where(test['avg_invalid_oda'] == 0.0, 0,
                               test[amount_column] / test['avg_invalid_oda'])

    test['reason_b_value'] = np.where(((test['customer_level'] == 'low') & (test['reason_b_value'] != 0.0)), 1 / test['reason_b_value'],
                               test['reason_b_value'])

    #train.loc[train['reason_b_value'].isnull(), 'reason_b_value'] = 0.0
    #test.loc[test['reason_b_value'].isnull(), 'reason_b_value'] = 0.0
    del train[id_column]
    del test[id_column]

    train.rename(columns={'temp0':id_column},inplace=True)
    test.rename(columns={'temp0':id_column}, inplace=True)   
    return train, test
