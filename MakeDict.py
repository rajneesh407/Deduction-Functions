model_dict={}
model_dict['Customer']=['fk_customer_map_id']
model_dict['dates']=['deduction_created_date']
model_dict['entity']=['fk_reason_code_id','payer','buyer name','ar_reason_code']
model_dict['amount']=['original_dispute_amount']


def getDictList(model_dict,key_name):
    return model_dict[key_name]



one=getDictList(model_dict,'entity')
print(one)
