def dispute_amount_deviation(history, left_history, train_239, test_239, customer_identifier, dispute_amount,
                             output_label='output_label', fill_null='others', customer_grouping_threshold=500,
                             regularized=False):
    # preprocessing of columns
    left_history[customer_identifier].fillna(fill_null, inplace=True)
    train_239[customer_identifier].fillna(fill_null, inplace=True)
    test_239[customer_identifier].fillna(fill_null, inplace=True)
    history[customer_identifier].fillna(fill_null, inplace=True)
    history[customer_identifier] = history[customer_identifier].astype(float).astype(int).astype(str)
    train_239[customer_identifier] = train_239[customer_identifier].astype(float).astype(int).astype(str)
    test_239[customer_identifier] = test_239[customer_identifier].astype(float).astype(int).astype(str)
    left_history[customer_identifier] = left_history[customer_identifier].astype(float).astype(int).astype(str)

    # grouping top customer for history data 
    all_values = history[customer_identifier].value_counts()
    top_values = all_values[all_values >= customer_grouping_threshold].index.tolist()
    print(top_values)
    history[customer_identifier] = get_group_column(history[customer_identifier], top_values)

    train_239[customer_identifier] = get_group_column(train_239[customer_identifier], top_values)
    test_239[customer_identifier] = get_group_column(test_239[customer_identifier], top_values)

    left_history[customer_identifier] = get_group_column(left_history[customer_identifier], top_values)
    # calculating average dispute amount per customer

    average_dispute_amount_per_customer = history.groupby(customer_identifier).agg({dispute_amount: 'mean'})
    average_dispute_amount_per_customer.reset_index(inplace=True)
    average_dispute_amount_per_customer.columns = [customer_identifier, 'average_dispute_amount']
    test_239 = pd.merge(test_239, average_dispute_amount_per_customer, on=customer_identifier, how='left')

    test_239['original_by_average_dispute_amount'] = test_239[dispute_amount] / test_239['average_dispute_amount']

    if regularized == True:

        y_train_239 = train_239[output_label]
        skf = model_selection.StratifiedKFold(5, shuffle=True, random_state=0)
        print(y_train_239.isnull().sum())
        for x_four_index, x_one_index in skf.split(train_239, y_train_239):
            train_239_fold, val_fold = train_239.iloc[x_four_index], train_239.iloc[x_one_index]
            train_239_fold = pd.concat([train_239_fold, left_history], axis=0)
            # finding avg_ODA
            average_dispute_amount_per_customer = train_239_fold.groupby(customer_identifier).agg({dispute_amount: 'mean'})
            average_dispute_amount_per_customer.reset_index(inplace=True)
            average_dispute_amount_per_customer.columns = [customer_identifier, 'average_dispute_amount_per_customer']
            val_fold = pd.merge(val_fold, average_dispute_amount_per_customer, on=customer_identifier, how='left')
            val_fold['average_dispute_amount_per_customer'].fillna(val_fold[dispute_amount], inplace=True)
            val_fold['original_by_average_dispute_amount'] = val_fold[dispute_amount] / val_fold[
                'average_dispute_amount_per_customer']
            train_239.loc[x_one_index, 'original_by_average_dispute_amount'] = val_fold[
                'original_by_average_dispute_amount'].values
    else:
        train_239 = pd.merge(train_239, average_dispute_amount_per_customer, on=customer_identifier, how='left')
        train_239['original_by_average_dispute_amount'] = train_239[dispute_amount] / train_239['average_dispute_amount']

    train_239['original_by_average_dispute_amount'].fillna(1, inplace=True)
    test_239['original_by_average_dispute_amount'].fillna(1, inplace=True)
    train_239['original_by_average_dispute_amount'] = train_239['original_by_average_dispute_amount'].astype(np.float64)
    test_239['original_by_average_dispute_amount'] = test_239['original_by_average_dispute_amount'].astype(np.float64)
    return train_239['original_by_average_dispute_amount'], test_239['original_by_average_dispute_amount'], average_dispute_amount_per_customer


