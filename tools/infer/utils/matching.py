import re
import json
import difflib
import numpy as np


from threading import Thread
from queue import Queue
from rapidfuzz import process, fuzz
from annoy import AnnoyIndex
from difflib import SequenceMatcher
from distutils import filelist
# from importlib.metadata import files
from unidecode import unidecode

class Processing():
    def __init__(self, dict_res, path_txt_noise_word = "models/noise_word.txt"):
        list_input = dict_res
        self.infos = list_input
        
        with open(path_txt_noise_word, encoding="utf8") as file:
            lines = file.readlines()
            list_noise = [line.rstrip() for line in lines]

        self.lst_noise = list_noise

    def noise_filter(self):
        textbox_list = []
        filter_list = []
        
        for idx,info in enumerate(self.infos[::]):
            name = info['transcription'].upper().strip()
            point = info['points']
            
            st = unidecode(name)
            # print(name)
            if re.search("^\d+\s?[A-z]{2,}", st) \
                    and name.find('/') == -1 \
                    or re.search("\d{5,}", st) \
                    or True in [char in self.lst_noise for char in name.split(' ')] \
                    or re.search("\d{1,2}[hH]", st) \
                    or re.search("CM$", st) \
                    or re.search("^HOT$", st):
                pass
            else:
                # Xoá dấu chấm
                st = unidecode(name)
                if re.search('(\.+)|(-\s)', st):
                    name = (re.sub('\.+|(-\s)','', name)).strip()
                
                ### Xoá địa chỉ
                st = unidecode(name)
                if re.search('^\d+[.,][A-z]+', st):
                    name = (re.sub('^\d+[.,]', '',name)).strip()
                #
                st = unidecode(name)
                if re.search('^\d+\.\d+$',st):
                    name = (re.sub('\.', '',name)).strip()
                
                # Đổi tiền 
                st = unidecode(name)
                st = re.sub(r'\s+','', st)
                check = 0
                if re.search(r"(?:\d+,)?(\d{1,})[/.]*", st):
                    res = re.findall(r"(?:\d+,)?(\d{1,})[/.]*", st)
                    check = 1
                    if len(res) != 1:
                        for name in res:
                            name = name if re.search('0{3,}', name) else name+'000'
                            filter_list.append(re.sub('[^\d]+','',name).strip())
                            textbox_list.append(point)
                    elif not re.search('0{3,}$', res[0]) :
                        if len(res[0]) > 3: 
                            for idx in range(0, len(res[0]), 2):
                                name = res[0][idx:idx+2]+'000'
                                filter_list.append(re.sub('[^\d]+','',name).strip())
                                textbox_list.append(point)
                        else:
                            if not re.search('[A-Z]+\d+[A-z]*', st) and not re.search('\d+[,-][^\d]+', st):
                                if not re.search('^([^\d]+\d+[A-z]+)|\.*%\.*', st):
                                    name = res[0] if re.search('0{3,}', res[0]) else res[0]+'000'
                                    # filter_list.append(name.strip())
                                    filter_list.append(re.sub('[^\d]+','',name).strip())
                                    textbox_list.append(point)
                                else:
                                    check=0
                            else:
                                check=0
                    else:
                        check=0

                if not check:
                    name = fix_char_name(name)
                    filter_list.append(name.strip())
                    textbox_list.append(point)
                
        return filter_list, textbox_list

### Difflib
class Post_processing():
    def __init__(self, path_dict_menu='models/ppocr_recog_dict.txt'):
        self.seq = SequenceMatcher()

    def __call__(self, input, lines, output):
        self.target = {}
        self.seq.set_seq1(input)
        
        for line in lines:
            self.seq.set_seq2(line)
            ratio = self.seq.ratio()
            # if ratio > 0.85:
            self.target[line] = ratio
            
        if len(self.target) != 0:
            Keymax = max(zip(self.target.values(), self.target.keys()))[1]
        else:
            Keymax = input

        output.put(Keymax.strip())

        return Keymax.strip()


### Rapid fuzz
class Post_processing_fuzzy():
    def __init__(self, path_dict_menu='models/ppocr_recog_dict.txt'):
        with open(path_dict_menu, 'rb') as fi:
            self.lines = fi.readlines()

        self.lines = [line.decode('utf-8')[:-1] for line in self.lines]

    def __call__(self, input, lst_fuzzy):
        self.res = self.getMatch(input)
        lst_fuzzy.put(self.res)
        return self.res
    
    def getMatch(self, input):
        fcs = [fuzz.QRatio,
            fuzz.token_ratio,
            fuzz.token_set_ratio,
            fuzz.partial_ratio,
            fuzz.partial_token_set_ratio,
            fuzz.partial_token_ratio,
            fuzz.WRatio,
            fuzz.partial_token_sort_ratio,
            fuzz.token_sort_ratio,
            fuzz.ratio]

        res = []
        for fc in fcs:
            temp = process.extractOne(input, self.lines,scorer = fc)
            if temp[1] > 60:
                res.append(temp[0])

        return res

my_fuzzy_char = Post_processing_fuzzy()
my_diff_char = Post_processing()

lst_fuzzy = Queue()
output = Queue()

def fix_food_name(input):
    Thread(target=my_fuzzy_char, args=(input, lst_fuzzy)).start()
    Thread(target=my_diff_char, args=(input, lst_fuzzy.get(), output)).start()

    return output.get()

def fix_char_name(input):
    Thread(target=my_fuzzy_char, args=(input, lst_fuzzy)).start()
    Thread(target=my_diff_char, args=(input, lst_fuzzy.get(), output)).start()

    return output.get()

def buildAnnoyIndex(data,metric="manhattan",ntrees=10):
    f = data.shape[1]
    idx = AnnoyIndex(f,metric)  

    for i,d in enumerate(data):
      idx.add_item(i, d)

    idx.build(ntrees) 

    return idx


def matching_row(dict_res):
    ###### Difflib ######  
    process = Processing(dict_res)

     ### noise filter ###
    filter_list, textbox_list= process.noise_filter()
    textbox_list = np.array(textbox_list)
    # print(filter_list)
    # print('--------------------------------')
    # print(textbox_list)

    ### Number list ###
    number_list = [[textbox, number]  for textbox, number in zip(textbox_list, filter_list) if re.search('^\d+[kK]?', number.strip())]
    number_list = sorted(number_list, key=lambda x:x[0][0,1])
    # print(number_list)

    #####################################################################
    ######## Find result_column ########
    result_column = []
    flag = 0
    for idx in range(len(number_list)):
        if idx + 1 >= len(number_list):
            # if number_list[idx][0][0,1] - number_list[idx-1][0][0,1] > 20: #20
            #     result_column.append(number_list[idx][0][:,-1])
            break
        
        if number_list[idx+1][0][0,1] - number_list[idx][0][0,1] > 20: #20
            if flag: 
                result_column.append(number_list[idx+1][0][:,-1])
            else: 
                result_column.append(number_list[idx][0][:,-1])
                result_column.append(number_list[idx+1][0][:,-1])
                flag = 1   

    # print(len(result_column))
    # print('-'*60)
    # print(result_column)

    ######## Find x_textbox_list, y_textbox_list ########
    x_textbox_list = textbox_list[:,:,0]
    y_textbox_list = textbox_list[:,:,1]
    # print('-'*60)
    # print(x_textbox_list[:5])
    # print('-'*60)
    # print(y_textbox_list[:5])

    #####################################################################
    threshsold_line = 15 #15
    Nb_neighbors = 20 #10
    #####################
    result_row = [] 

    for output_col in result_column:
        output_row = []
        vector = output_col

        annoy_idx = buildAnnoyIndex(y_textbox_list)
        bucket_idx = annoy_idx.get_nns_by_vector(vector,Nb_neighbors)
        for idx in bucket_idx:
            # print(textbox_list[idx], filter_list[idx])
            if abs(y_textbox_list[idx][0] - vector[0]) <= threshsold_line:
                output_row.append([textbox_list[idx], filter_list[idx]])
        output_row = sorted(output_row, key=lambda x: x[0][0,0])
        result_row.append(output_row)

    # print(len(result_row))
    # print(result_row[-5])
    # print('-'*60)
    # print(result_row[-2])
    #####################################################################
    results = []
    size = [[' M', ' L'],[' S', ' M', ' L'], [' S', ' M', ' L', ' XL'],\
            [' S', ' M', ' L', ' XL', ' XXL'], \
            [' S', ' M', ' L', ' XL', ' XXL', " XXXL"]]
    for output_row in result_row:
        temp_price = []
        food = ''
        for idx, infos in enumerate(output_row):
            if not re.search('^\d{2,}', infos[1]):
                if len(temp_price) != 0 and food != '':
                    if len(temp_price) > 1:
                        for i in range(len(temp_price)):
                            # fixed_food = fix_food_name(food.strip()+size[len(temp_price)-2][i])
                            fixed_food = food.strip()+size[len(temp_price)-2][i]
                            results.append([fixed_food, int(temp_price[i])])
                    else:
                        # food = fix_food_name(food)
                        results.append([food.strip(), int(temp_price[0])])
                    temp_price = []
                    food = ''
                food += ' ' + infos[1]
            else:
                temp_price.append(infos[1])
                
            if idx==len(output_row)-1 and len(temp_price) != 0 and food != '':
                if len(temp_price) > 1:                
                    for i in range(len(temp_price)):
                        # fixed_food = fix_food_name(food.strip()+size[len(temp_price)-2][i])
                        fixed_food = food.strip()+size[len(temp_price)-2][i]
                        results.append([fixed_food, int(temp_price[i])])
                else:
                    # food = fix_food_name(food)
                    results.append([food.strip(), int(temp_price[0])])
    
    # for result in results:
    #     print(result)
    return results

# if __name__ == "__main__":
#     results = matching_row(5)
#     for result in results:
#         print(result)