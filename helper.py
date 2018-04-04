import re
import jieba
import jieba.analyse
jieba.load_userdict("user_dict.txt")
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import word2vec
from langconv import *
from difflib import SequenceMatcher
import math
from scipy.stats import kurtosis
from scipy.stats import skew



# read stopword list
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords 

stopword_list = stopwordslist('stopwords.txt')

# clean string and split words, output is a list of words
def split_words(text, mode, stopwords = [' '] + stopword_list):
    
    # replace emoji
    try:
        # UCS-4
        highpoints = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])|([\U000025A0-\U000025FF])|([\U00002B50])|([\U000010E6])')
    except re.error:
        # UCS-2
        highpoints = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])|([\u25A0-\u25FF])|([\u2B50])|([\u10e6])')
    
    res = highpoints.sub(u'??', text)  
    
    # for english character: convert to lower case
    res = res.lower()
    
    # split words
    word_list = jieba.lcut_for_search(res, HMM=True)
    
    # remove text within parentheses
    #res = re.sub(r'\([^)]*\)', '', text)
    
    # remove punctuation
    if mode == 'simplified':
        word_list = [Converter('zh-hans').convert(ele) for ele in word_list if ele not in stopwords]
    elif mode == 'traditional':
        word_list = [ele for ele in word_list if ele not in stopwords]
		
    # remove substring
    delete_words = []
    for idx in range(len(word_list)):
        for i in range((idx+1), len(word_list)):
            if ((word_list[idx] in word_list[i]) & (word_list[idx] != word_list[i])):
                delete_words.append(word_list[idx]) 
    word_list = [word for word in word_list if word not in delete_words] 
    
    if len(word_list) < 1:
        word_list = [text.strip()]
    
    return word_list

# clean string and split words, output is a string and words are speperated by space
def pre_processing(text, mode, stopwords = [' '] + stopword_list):
    
    # replace emoji
    try:
        # UCS-4
        highpoints = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])|([\U000025A0-\U000025FF])|([\U00002B50])|([\U000010E6])')
    except re.error:
        # UCS-2
        highpoints = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])|([\u25A0-\u25FF])|([\u2B50])|([\u10e6])')
    
    res = highpoints.sub(u'??', text)  
    
    # for english character: convert to lower case
    res = res.lower()
    
    # split words
    word_list = jieba.lcut_for_search(res, HMM=True)
    
    # remove punctuation
    #res = re.sub(u'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）|)(．ღ＊《》『』【】<>▲▶◀◎◆◤◥[]～-]+', '',text)  
    if mode == 'simplified':
        word_list = [Converter('zh-hans').convert(ele) for ele in word_list if ele not in stopwords]
    elif mode == 'traditional':
        word_list = [ele for ele in word_list if ele not in stopwords]
    
    # remove substring
    delete_words = []
    for idx in range(len(word_list)):
        for i in range((idx+1), len(word_list)):
            if ((word_list[idx] in word_list[i]) & (word_list[idx] != word_list[i])):
                delete_words.append(word_list[idx]) 
    word_list = [word for word in word_list if word not in delete_words] 
    
    # remove stopwords
    if len(word_list) < 1:
        word_list = text.strip()
    else:
        text = ' '.join(word_list)
    
    return text
	
	
def calculate_betweenss(centroids):
    b = 0
    for i in range(len(centroids)): 
        for j in range(i+1, len(centroids)):
            dis = float(cosine_similarity(centroids[i], centroids[j]))
            b += dis
    return b

# read a word2vec model
def read_word2vec(file_name):
    model = word2vec.Word2Vec.load(file_name + '.model')
    keys=list(model.wv.vocab.keys())
    
    wordvector=[]
    for key in keys:
        wordvector.append(model[key])
        
    return model, keys, wordvector

# calculate similarity between words
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
	
# vectorize strings
def vectorizer(text, model, keys, wordvector):
        
    words = text.strip().split()
    
    lis = []
    
	# vectorize each word in string
    for word in words:
        if word in keys:
            vec = model[word]
        else:
            max_sim_score = max([similar(word, key) for key in keys])
            vec = np.mean([wordvector[idx] for idx, key in enumerate(keys) if similar(word, key) == max_sim_score][0:3], axis = 0)
        
        lis.append(vec)
		
    # aggregate all wordvectors occur in a string
    min_v = list(np.min(lis, axis = 0))
    mean_v = list(np.mean(lis, axis = 0))
    max_v = list(np.max(lis, axis = 0))
    kurtosis_v = list(kurtosis(lis))
    skewness_v = list(skew(lis))

    return np.array(min_v + mean_v + max_v + kurtosis_v + skewness_v)

	
def data_prep(df):
    
    stopword_list = stopwordslist('stopwords.txt')
    
    df['Product_Name_s'] = df.apply(lambda row: pre_processing(row['Product Name'], mode = 'simplified'), axis = 1)
    df['Product_Name_t'] = df.apply(lambda row: pre_processing(row['Product Name'], mode = 'traditional'), axis = 1)
    df['Query_s'] = df.apply(lambda row: pre_processing(row['Query'], mode = 'simplified'), axis = 1)
    df['Query_t'] = df.apply(lambda row: pre_processing(row['Query'], mode = 'traditional'), axis = 1)
    
    return df

# get summary data about the occurence of a keyword across titles
def get_pos_info(df, key):
    pos_list = []
    n = 0
    for name in df['Product_Name_s'].tolist():
        n += 1
        if key in name.strip().split():
            pos_list.append((name.strip().split().index(key)+1)/len(name.strip().split()))
    if len(pos_list) < 1:
        pos_list = [0]
    min_pos, max_pos, mean_pos, idf = min(pos_list), max(pos_list), sum(pos_list)/len(pos_list), math.log(n/(len(pos_list)+1))
    return min_pos, max_pos, mean_pos, idf
	
# get a dictionary storing all the summary of occurence of all keywords across all titles
def get_word_info(df, w2vmodel):
    word_info = {}
    model, keys, wordvector = read_word2vec(w2vmodel)
    for key in keys:
        min_pos, max_pos, mean_pos, idf = get_pos_info(df, key)
        word_info[key] = np.array((min_pos, max_pos, mean_pos, idf))
    return word_info

# define a function for creating new features for supervised learning
def find_keyword_info(query, name, res, keys, word_info):
    query_words = query.strip().split()
    name_words = name.strip().split()
    pos_list = []
    info = np.zeros((4, ))
    for word in query_words:
        if word in name_words:
            pos_list.append(name_words.index(word)+1)
            if word in keys:
                info += word_info[word]
            else:
                max_sim_score = max([similar(word, w) for w in name_words])
                if max_sim_score > 0.5:
                    try:
                        info += np.mean(np.array([word_info[w] for w in name_words if similar(word, w) == max_sim_score]))
                    except:
                        pass
            
        else:
            max_sim_score = max([similar(word, w) for w in name_words])
            if max_sim_score >= 0.5:
                try:
                    pos_list += [(name_words.index(w)+1) for w in name_words if similar(word, w) == max_sim_score]
                    info += np.mean(np.array([word_info[w] for w in name_words if similar(word, w) == max_sim_score]))
                except:
                    pass
            else:
                try:
                    pos_list.append((name_words.index(model.most_similar(word)[0][0]) + 1))
                    info += np.mean(word_info[model.most_similar(word)[0][0]])
                except:
                    pass
    if len(pos_list) < 1:
        pos_list = [0]
        
    if res == 'min_pos':
        return_res = min(pos_list)
    elif res == 'max_pos':
        return_res =  max(pos_list)
    elif res == 'mean_pos':
        return_res = sum(pos_list)/ len(pos_list)
    elif res == 'tf':
        return_res = len(pos_list)
    elif res == 'tmin_pos':
        return_res = info[0]
    elif res == 'tmax_pos':
        return_res = info[1]
    elif res == 'tmean_pos':
        return_res = info[2]
    elif res == 'idf':
        return_res = info[3]
    return return_res


	
	
