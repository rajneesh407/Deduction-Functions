import xlrd
import operator

def individualInteraction(first_sheet):
      df={}
      for i in range(7):
          df_temp = {}
          for row in range(1,first_sheet.nrows):
              df_temp[first_sheet.row_values(row)[0]]=first_sheet.row_values(row)[7+i]
          df_temp_soted=sorted(df_temp.items(),  key=lambda x: x[1])

          temp_list=[i[0] for i in df_temp_soted]
          df[first_sheet.row_values(0)[7+i]]=temp_list
      return df




def  oneLevelHistory(second_sheet):
     df = {}
     for i in range(7):
         df_temp = {}
         for row in range(1, second_sheet.nrows):
             if (second_sheet.row_values(row)[0].split("|")[0]) != (second_sheet.row_values(row)[0].split("|")[1]):
              df_temp[second_sheet.row_values(row)[0]] = second_sheet.row_values(row)[7 + i]
         df_temp_soted = sorted(df_temp.items(), key=lambda x: x[1])

         temp_list = [i[0] for i in df_temp_soted]
         #print(temp_list)
         df[second_sheet.row_values(0)[7 + i]] = temp_list

     return df


def twoLevelHIstory(third_sheet):
    df = {}
    for i in range(7):
        df_temp = {}
        for row in range(1, third_sheet.nrows):
            if ((third_sheet.row_values(row)[0].split("|")[0]) != (third_sheet.row_values(row)[0].split("|")[1])
                and (third_sheet.row_values(row)[0].split("|")[1]) != (third_sheet.row_values(row)[0].split("|")[2])
                and (third_sheet.row_values(row)[0].split("|")[2]) != (third_sheet.row_values(row)[0].split("|")[0])
                ):
                df_temp[third_sheet.row_values(row)[0]] = third_sheet.row_values(row)[7 + i]
        df_temp_soted = sorted(df_temp.items(), key=lambda x: x[1])

        temp_list = [i[0] for i in df_temp_soted]
        df[third_sheet.row_values(0)[7 + i]] = temp_list
    return df


def interactions(path,individual_rank,onelevel_rank,twolevel_rank):

    workbook=xlrd.open_workbook(path)
    indivual_interaction=individualInteraction(workbook.sheet_by_index(0))
    one_level_interaction=oneLevelHistory(workbook.sheet_by_index(1))
    two_level_interaction=twoLevelHIstory(workbook.sheet_by_index(2))

    length_one=len(one_level_interaction[onelevel_rank])
    one_level=[]
    for i in range(length_one):
        x=[]
        x.append(one_level_interaction[onelevel_rank][i].split("|")[0])
        x.append(one_level_interaction[onelevel_rank][i].split("|")[1])
        one_level.append(x)

    two_level = []
    length_two=len(two_level_interaction[twolevel_rank])
    for i in range(length_two):
        x = []
        x.append(two_level_interaction[twolevel_rank][i].split("|")[0])
        x.append(two_level_interaction[twolevel_rank][i].split("|")[1])
        x.append(two_level_interaction[twolevel_rank][i].split("|")[1])

        two_level.append(x)

    #print(one_level)
    #print(two_level)
    return indivual_interaction[individual_rank],one_level,two_level


path=r"C:\Users\rajneesh.jha\Downloads\deduction\XGB\preliminary_model.xlsx"
one,two,three=interactions(path,'Average Rank','Gain Rank','FScore Rank')
print(two)
print(three)
def column_converter(li):
    string = '_history'
    for i in range(len(li)):
        for j in range(len(li[i])):
            if string in li[i][j]:
                li[i][j] = li[i][j][:li[i][j].index('_history')]
    return li
two = column_converter(two)
three = column_converter(three)
print(one)

print(two)
print(three)
