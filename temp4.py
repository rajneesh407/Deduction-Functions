l=['fk_reason_code_map_id_history', 'payer_history', 'ar_reason_code_history', 'sold_to_party_history', 'ar_reason_code_sold_to_party_rank', 'payer_sold_to_party_rank', 'fk_reason_code_map_id_sold_to_party_rank', 'fk_reason_code_map_id_payer_rank', 'ar_reason_code_sold_to_party_history', 'payer_sold_to_party_history', 'fk_reason_code_map_id_sold_to_party_history', 'fk_reason_code_map_id_payer_history', 'b_value', 'row_history', 'r_value']
def getKnnList():
    matching = [s for s in l if "_history" in s]
    return matching
knn_list=getKnnList()
print(knn_list)