from collections import Counter, defaultdict
import json
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
import sys
from scipy import sparse

from emoji_function import demojize

def config_matplotlib():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure t
    import seaborn as sns
    sns.set(style="whitegrid")
    sns.set_context("paper", rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
                                 "xtick.labelsize": 14, "ytick.labelsize": 14})


def load_posts(data_file, label_file):
    labels = [json.loads(l) for l in open(label_file)]
    posts = []
    for l in open(data_file, encoding="utf8"):
        l = json.loads(l)
        post = {}
        post['code'] = l['code']
        post['create_at'] = l['create_at']
        post['comments'] = []
        post['users'] = []
        post['labels'] = []
        post['timestamps'] = []
        label_d = [x for x in labels if x['code'] == l['code']][0]['label']
        for comment_n in sorted(l['text'].keys(), key=lambda x: int(x)):
            post['comments'].append(l['text'][comment_n])
            post['users'].append(l['username'][comment_n])
            post['labels'].append(label_d[comment_n])
            post['timestamps'].append(l['timestamp'][comment_n])

        post['num_hostile'] = len([x for x in post['labels'] if x != 'Innocuous'])
        posts.append(post)
    return np.array(posts)

def filter_by_num_hostile(posts, max_for_neg_class=1, min_for_pos_class=10):
    return [i for i,p in enumerate(posts) if
            p['num_hostile'] <= max_for_neg_class or
            p['num_hostile'] >= min_for_pos_class
           ]

def get_hostile_indices(post):
    return [i for i,v in enumerate(post['labels']) if v != 'Innocuous']


def set_observed_comments(posts, lead_time=10*60*60):
    """
    Assumes we already have pairs of pos/neg posts.
    Modifies the n_comments_observed fields to match the given lead_time.
    """
    pos_posts = [p for p in posts if p['num_hostile'] > 0]
    neg_posts = [p for p in posts if p['num_hostile'] == 0]
    pos_posts = sorted(pos_posts, key=lambda p: p['n_comments_observed'])
    neg_posts = sorted(neg_posts, key=lambda p: p['n_comments_observed'])
    for pos, neg in zip(pos_posts, neg_posts):
        hostile_idx = get_hostile_indices(pos)[0]
        hostile_time = pos['timestamps'][hostile_idx]
        stop_time = hostile_time - lead_time
        num_comments = len([i for i in pos['timestamps'] if i < stop_time])
        pos['n_comments_observed'] = neg['n_comments_observed'] = num_comments
    return pos_posts + neg_posts

def sample_posts_task1(posts, lead_time=10*60*60):
    """
    lead_time....minimum number of seconds between final observed comment and first observed hostile comment.
    """
    for p in posts:
        p['n_comments_observed'] = 0
    # for positive posts
    n_comments_obs = []
    positive_posts = [p for p in posts if p['num_hostile'] > 0]
    valid_positive_posts = []
    for p in positive_posts:
        hostile_idx = get_hostile_indices(p)[0]
        hostile_time = p['timestamps'][hostile_idx]
        stop_time = hostile_time - lead_time
        num_comments = len([i for i in p['timestamps'] if i < stop_time])
        n_comments_obs.append(num_comments)
        p['n_comments_observed'] = num_comments
        if num_comments > 0:
            valid_positive_posts.append(p)
    print('total non-zero posts %d' % len([k for k in n_comments_obs if k > 0]))
    # for negative posts, sample same distribution of comment counts.
    negative_posts = [p for p in posts if p['num_hostile'] == 0]
    indices = np.arange(len(negative_posts))
    valid_positive_posts = sorted(valid_positive_posts, key=lambda p: p['n_comments_observed'], reverse=True)
    negative_posts = sorted(negative_posts, key=lambda p: len(p['comments']), reverse=True)
    sampled_negative_posts = []
    sampled_positive_posts = []
    while len(valid_positive_posts) > 0:
        pos_post = valid_positive_posts.pop()
        n = pos_post['n_comments_observed']
        while True and len(negative_posts) > 0:
            p = negative_posts.pop()
            if len(p['comments']) > n:
                p['n_comments_observed'] = n
                sampled_negative_posts.append(p)
                sampled_positive_posts.append(pos_post)
                break
    print('sampled %d negative posts' % len(sampled_negative_posts))
    print('sampled %d positive posts' % len(sampled_positive_posts))
    return sampled_positive_posts + sampled_negative_posts

def set_n_comments_observed_task2(posts):
    """
    Observe comments up to and including first hostile.
    """
    for p in posts:
        p['n_comments_observed'] = get_hostile_indices(p)[0] + 1

def vectorize(posts):
    def yield_first_comments(posts):
        for p in posts:
            yield p['comments'][:p['n_comments_observed']]
    vec = CountVectorizer(min_df=2, ngram_range=(1,1))
    clean_texts = [cleanText(' '.join(c)) for c in yield_first_comments(posts)]
    X_unigram = vec.fit_transform(clean_texts)
    X_hatebase, header_hatebase = hatebase(clean_texts)
    X_profane, header_profane = profaneLexicon(clean_texts)
    X_aggregation_letters, header_w2v_char = w2v_aggregation_letters(clean_texts)
    X_aggregation, header_w2v = w2v_aggregation(clean_texts)
    X_previous_post, header_previous_post = previous_post_features([p['code'] for p in posts])
    X_previous_comment, header_previous_comment = previous_comment_features(posts)  ###
    X_trend, header_trend = trend_features([c for c in yield_first_comments(posts)])
    X_user, header_user = user_features(posts)  ###
    X_final_comment, header_final_comment = final_comment_features([cleanText(list(c)[-1]) for c in yield_first_comments(posts)])
    header_vec = [vec.get_feature_names(), header_hatebase + header_profane, header_w2v_char, header_w2v,
                 header_previous_post, header_previous_comment, header_trend, header_user, header_final_comment]
    X = sparse.hstack([X_unigram, X_hatebase, X_profane, X_aggregation_letters, X_aggregation,
                       X_previous_post, X_previous_comment, X_trend, X_user, X_final_comment]).tocsr()
    return X, vec, header_vec, ['Unigram', 'lex', 'n-w2v', 'w2v', 'prev-post', 'prev-com', 'trend', 'user', 'final-com']

def cleanText(text):
    
    text = bytes(text, 'utf-8','ignore').decode('utf-8','ignore')
    # Add " emoji_" as prefix
    text = demojize(text)  
    # Add " hashtag_" as prefix
    text = re.sub(r"#(\w+)", r" hashtag_\1 ", text)
    # Substitute mentions with specialmentioned
    text = re.sub('@', ' @', text)
    text = re.sub('(?:^|[^\w])(?:@)([A-Za-z0-9_](?:(?:[A-Za-z0-9_]|(?:\.(?!\.))){0,28}(?:[A-Za-z0-9_]))?)', 
                  ' specialmentioned ', text)
    # Substitute urls with specialurl
    text = re.sub('http\S+', ' specialurl ', text)
    # Remove other symbols
    text = re.sub(" '|'\W|[-(),.\"!?#*$~`\{\}\[\]/+&*=:^]", " ", text)
    text = re.sub("\s+", " ", text).lower().strip().split()
    # Remove repetition letters
    text_list = [replaceThreeOrMore(i) for i in text] 
    
    result = " ".join(text_list)
    
    if result == " " or result == "":
        return "blank_comment"
    else:
        return result


def replaceThreeOrMore(word):
    """
    look for 3 or more repetitions of letters and replace with this letter itself only once
    """
    pattern = re.compile(r"(.)\1{3,}", re.DOTALL)
    return pattern.sub(r"\1", word)



def cleanText_letters(text):
    """
    replace emoji/hashtag/specialmentioned/specialurl to blank space
    """
    text = re.sub(r"emoji_(\w+)", r" ", text)
    text = re.sub(r"hashtag_(\w+)", r" ", text)
    text = re.sub(r"specialmentioned", r" ", text)
    text = re.sub(r"specialurl", r" ", text)
    text = re.sub("\s+", " ", text).lower().strip()   

    if text == " " or text == "":
        return "blank_comment"
    else:
        return text    
    
    return text

def cv(X, y, n_splits=10, n_comments=[]):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    res = defaultdict(list)
    res_by_n_comments = []
    for train, test in kf.split(X):
        clf = LogisticRegression(class_weight='balanced')
        clf.fit(X[train], y[train])
        preds = clf.predict(X[test])
        probas = clf.predict_proba(X[test])[:,1]
        truths = y[test]
        if len(n_comments) > 0:
            res_by_n_comments.extend(zip(truths, probas, n_comments[test]))
        res['AUC'].append(roc_auc_score(truths, probas, average=None))
        res['F1'].append(f1_score(truths, preds, average=None))
        res['Recall'].append(recall_score(truths, preds, average=None)[1])
        res['Precision'].append(precision_score(truths, preds, average=None)[1])
        res['Accuracy'].append(accuracy_score(truths, preds))
    res_mean = {}
    for k,v in res.items():
        res_mean[k] = np.mean(v)
        res_mean['%s_sd' % k] = np.std(v)
    res_mean['by_n_comments'] = res_by_n_comments
    return res_mean

def has_mentioned(x):
    
    x_cleaned = cleanText(x)
    return "specialmentioned" in x_cleaned

def user_features(posts):
    
    """
    1. Ratio of users in the conversation
    2. Ratio of "directed" comments (those with mentions)
    """
    
    result = []
    for post in posts:
        #n_comments = get_hostile_indices(post)[0] + 1
        n_comments = post['n_comments_observed']
        feature_list = []
        num_mentioned = 0
        user_list = post['users'][:n_comments]
        comment_list = post['comments'][:n_comments]
        ratio_users = len(set(user_list))/len(user_list)
        for c in comment_list:
            flag = has_mentioned(c)            
            if flag:
                num_mentioned += 1
                    
        ratio_mentioned = num_mentioned/len(user_list)
                
        feature_list.append(ratio_users)
        feature_list.append(ratio_mentioned)
            
        result.append(feature_list)

    X_matrix = np.asarray(result)
    headers = ["other_features_1st", "other_features_2st"]
        
    return X_matrix, headers



def hatebase(X_raw, hatebasepath=os.path.join(os.environ['NOBULL_PATH'], 'hatebase.json')):
    '''
    The hatabase features has length 14. 
    1.  Binary representation of occurrence
    2.  Binary representation of about_class
    3.  Binary representation of about_disability
    4.  Binary representation of about_ethnicity
    5.  Binary representation of about_gender
    6.  Binary representation of about_nationality
    7.  Binary representation of about_religion
    8.  Aggregation of occurrence (sum)
    9.  Aggregation of about_class 
    10. Aggregation of about_disability 
    11. Aggregation of about_ethnicity 
    12. Aggregation of about_gender 
    13. Aggregation of about_nationality 
    14. Aggregation of about_religion 
    '''
    
    with open(hatebasepath, 'r') as fp:
        vocab = json.load(fp)
    
    num_features = 14
    num_row = len(X_raw)
    
    header = []
    
    for i in range(num_features):
        temp_string = "hatebase_" + str(i) + "-th"
        header.append(temp_string)
        
    hatebase_matrix = np.zeros(shape=(num_row, num_features))
    
    for rowIndex in range(num_row):
        temp_text = X_raw[rowIndex]
        temp_token_list = temp_text.split()
        num_occur = num_class = num_disability = num_ethnicity = num_gender = num_nationality = num_religion = 0
        
        current_aggreg_V = [0]*7
        
        for token in temp_token_list:
            if token in vocab:
                current_aggreg_V[0] += 1
                current_aggreg_V[1] += int(vocab[token]['about_class'])
                current_aggreg_V[2] += int(vocab[token]['about_disability'])
                current_aggreg_V[3] += int(vocab[token]['about_ethnicity'])
                current_aggreg_V[4] += int(vocab[token]['about_gender'])
                current_aggreg_V[5] += int(vocab[token]['about_nationality'])
                current_aggreg_V[6] += int(vocab[token]['about_religion'])
                
        current_binary_V = [1 if x >= 1 else x for x in current_aggreg_V]
        result_V = current_binary_V + current_aggreg_V
        hatebase_matrix[rowIndex] = np.array(result_V)

    return hatebase_matrix, header


def profaneLexicon(X_raw, lexiconpath=os.path.join(os.environ['NOBULL_PATH'], 'profanity.txt')):
    '''
    profaneLexicon has length 2
    1. Binary representation
    2. Aggregation representation
    '''
    lexicon_list = []
    
    with open(lexiconpath, 'r') as fp:
        for word in fp.readlines():
            lexicon_list.append(word.strip())
            
    num_features = 2
    num_row = len(X_raw)

    header = []
    
    for i in range(num_features):
        temp_string = "profaneLexicon_" + str(i) + "-th"
        header.append(temp_string)
    
    profaneLexicon_matrix = np.zeros(shape=(num_row, num_features))
    
    for rowIndex in range(num_row):
        temp_text = X_raw[rowIndex]
        temp_token_list = temp_text.split()
        
        current_aggreg_value = [0]
        
        for token in temp_token_list:
            if token in lexicon_list:
                current_aggreg_value[0] += 1
            
        current_binary_value = [1 if x >= 1 else x for x in current_aggreg_value]
        result_V = current_binary_value + current_aggreg_value
        profaneLexicon_matrix[rowIndex] = np.array(result_V)
        
    return profaneLexicon_matrix, header

#w2v_model = Word2Vec.load(os.path.join(os.environ['NOBULL_PATH'], 'model.w2v'))
#w2v_model_3gram = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.environ['NOBULL_PATH'], 'w2v_char.vec'))
w2v_model = None
w2v_model_3gram = None

def w2v_aggregation(X, length_vector=100):
    """
    First 100 dimention is the average of w2v vectors
    Second 100 dimention is the maximum of w2v vectors
    """
    global w2v_model
    if w2v_model == None: # lazy load
        w2v_model = Word2Vec.load(os.path.join(os.environ['NOBULL_PATH'], 'model.w2v'))
    num_row = len(X)

    max_matrix = np.zeros(shape=(num_row, length_vector))
    average_matrix = np.zeros(shape=(num_row, length_vector))

    for row in range(num_row):
        
        temp_text = X[row]    
        temp_vector = temp_text.split()
        
        unique_vector = list(set(temp_vector))
        num_index = len(unique_vector)
                    
        temp_matrix = np.zeros(shape=(num_index, length_vector))
        
        j = 0
        for word in unique_vector:
            try:
                temp_matrix[j] = w2v_model[word]
            except:
                temp_matrix[j] = np.zeros(shape=(length_vector,))
            j += 1

        max_matrix[row] = np.maximum.reduce(temp_matrix)
        average_matrix[row] = np.mean(temp_matrix, axis=0)
        
    result = np.concatenate((average_matrix, max_matrix), axis=1)
    result = sparse.csr_matrix(result)
    
    header = []
    
    for i in range(length_vector):
        temp_string = "oldw2v_average_" + str(i) + "-th"
        header.append(temp_string)
        
    for i in range(length_vector):
        temp_string = "oldw2v_maximum_" + str(i) + "-th"
        header.append(temp_string)    

    return result, header



def w2v_aggregation_letters(X, length_vector=100):
    """
    First 100 dimention is the average of w2v vectors
    Second 100 dimention is the maximum of w2v vectors
    """
    global w2v_model_3gram
    if w2v_model_3gram == None:
        w2v_model_3gram = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.environ['NOBULL_PATH'], 'w2v_char.vec'))
    X_raw = []
    for x in X:
        x_letter = cleanText_letters(x)
        X_raw.append(x_letter)


    num_row = len(X_raw)

    max_matrix = np.zeros(shape=(num_row, length_vector))

    average_matrix = np.zeros(shape=(num_row, length_vector))

    for row in range(num_row):
        
        temp_text = X_raw[row]    
        temp_vector = temp_text.split()
        
        unique_vector = list(set(temp_vector))
        num_index = len(unique_vector)
                    
        temp_matrix = np.zeros(shape=(num_index, length_vector))
        
        j = 0
        for word in unique_vector:
            
            temp_matrix[j] = get_vector(word, w2v_model_3gram, 100)
            j += 1

        max_matrix[row] = np.maximum.reduce(temp_matrix)
        average_matrix[row] = np.mean(temp_matrix, axis=0)
        
    result = np.concatenate((average_matrix, max_matrix), axis=1)
    result = sparse.csr_matrix(result)
    
    header = []
    
    for i in range(length_vector):
        temp_string = "neww2v_average_" + str(i) + "-th"
        header.append(temp_string)
        
    for i in range(length_vector):
        temp_string = "neww2v_maximum_" + str(i) + "-th"
        header.append(temp_string)

    return result, header


def get_vector(word, w2v_model, length_vector):
    
    if word == "blank_comment":
        return np.zeros(shape=(length_vector, ))
    else:
        try:
            vector = w2v_model[word]
        except:
            vector = np.zeros(shape=(length_vector, ))

        return vector
    
def previous_post_features(code_list, previous_post_file=os.path.join(os.environ['NOBULL_PATH'], 'code2previouspost.json')):
    
    """
    Previous posts from the owner of current post 
    """
    with open(previous_post_file, "r", encoding="utf8") as f:
        d = f.readlines()[0]
        code2previouspost_map = json.loads(d)
    
    X_previous_comments = []
    
    for c in code_list:
        if c in code2previouspost_map:
            previouspost = code2previouspost_map[c][1]
            text_lists = previouspost["text"]
            int_text = {int(k):v for k, v in text_lists.items()}
            
            current_X_text_list = []
            for key, value in sorted(int_text.items()):
                current_X_text_list.append(value)
            
            current_X_text = " ".join(current_X_text_list) 
            cleaned_text = cleanText(current_X_text)
            X_previous_comments.append(cleaned_text)
        else:
            X_previous_comments.append("blank_comment")

    X_hatebase, header_hatebase = hatebase(X_previous_comments)
    X_profane, header_profane = profaneLexicon(X_previous_comments)
    X_aggregation, header_w2v = w2v_aggregation_letters(X_previous_comments)
    
    vec = CountVectorizer(min_df=2)
    X = vec.fit_transform(X_previous_comments)
    
    X_overall = sparse.hstack([X, X_aggregation, X_hatebase, X_profane]).tocsr()
    
    header_vec = vec.get_feature_names()
    header = header_vec + header_w2v + header_hatebase + header_profane
    
    header_ = ['pre_post_'+ h  for h in header]
    
    return X_overall, header_


def previous_comment_features(posts, previous_comment_file=os.path.join(os.environ['NOBULL_PATH'], 'user2comment.json'), max_per_user=10):
    
    """
    Previous comments from users on this post.
    Since this can generate many features, we put a limit on the number of comments per user we consider.
    """
    with open(previous_comment_file, "r", encoding="utf8") as f:
        d = f.readlines()[0]
        user2previouscomment_map = json.loads(d)
        
    X_previous_comments = []
    posts_dict = {}
    comments_added = []
    for post in posts:
        #n_comment = get_hostile_indices(post)[0] + 1
        n_comment = post['n_comments_observed']
        create_at = post["create_at"]
        userlist = post["users"][:n_comment]
        current_X_text_list = []
        for user in userlist:
            if user in user2previouscomment_map:
                previouscomment = user2previouscomment_map[user]
                int_key = {float(k):v for k, v in previouscomment.items()}
                n_per_user = 0
                for ts, tx in sorted(int_key.items(), reverse = True):
                    if ts < int(create_at):
                        current_X_text_list.append(tx)
                        n_per_user += 1
                        if n_per_user > max_per_user:
                            break
                        
        if len(current_X_text_list) == 0:
            current_X_text_list.append('blank_comment')
        comments_added.append(len(current_X_text_list))
        current_X_text = " ".join(current_X_text_list) 
        cleaned_text = cleanText(current_X_text)
        X_previous_comments.append(cleaned_text)          
    X_hatebase, header_hatebase = hatebase(X_previous_comments)
    X_profane, header_profane = profaneLexicon(X_previous_comments)
    X_aggregation, header_w2v = w2v_aggregation_letters(X_previous_comments)
    
    vec = CountVectorizer(min_df=2)
    X = vec.fit_transform(X_previous_comments)
    
    X_overall = sparse.hstack([X, X_aggregation, X_hatebase, X_profane]).tocsr()
    
    header_vec = vec.get_feature_names()
    header = header_vec + header_w2v + header_hatebase + header_profane
    header_ = ['pre_c_'+ h  for h in header]
    return X_overall, header_

def final_comment_features(X_raw):
    """
    Distinguish between features from most recent comment.
    """
    vec = CountVectorizer(min_df=2)
    X = vec.fit_transform(X_raw)
    X_aggregation, header_w2v = w2v_aggregation_letters(X_raw)
    X_hatebase, header_hatebase = hatebase(X_raw)
    X_profane, header_profane = profaneLexicon(X_raw)
    X_overall = sparse.hstack([X, X_aggregation, X_hatebase, X_profane]).tocsr()
    header_vec = vec.get_feature_names()
    header = header_vec + header_w2v + header_hatebase + header_profane
    header_ = ['final_comment_'+ h  for h in header]
    return X_overall, header_

def trend_features(X_raw):
    '''
    1. Maximum value of probability of class 1 (positive)
    2. # of comments which have class 1 probability larger than 0.3
    3. Maximum slope within 2 adjacent probas in a list of probability of class 1
    4. Difference of max and min probability of class 1
    5. Ratio of feature 2
    '''
    
    row = len(X_raw)
    col = 4
    
    with open(os.path.join(os.environ['NOBULL_PATH'], 'clf&vec.pickle'), 'rb') as handle:
        pretrained_model, vectorizer = pickle.load(handle)
    
    result = np.zeros(shape=(row, col))
    
    maximum_1 = []
    num_comments_2 = []
    max_slope_3 = []
    max_difference_4 = []
    ratio_5 = []
    
    for i in X_raw:
        
        X_v_current_post = vectorizer.transform(i)
        
        X_letters = []
        for x in i:
            x_letter = cleanText_letters(x)
            if x_letter == " " or x_letter == "":
                X_letters.append("blank_comment")
            else:
                X_letters.append(x_letter)
        
        # Hatebase features
        X_hatebase, h_hate = hatebase(i)  
        # ProfaneLexicon
        X_profane, h_profane = profaneLexicon(i)
        # W2V
        X_letters_w2v_features, h_neww2v = w2v_aggregation_letters(X_letters)
        
        X = sparse.hstack([X_v_current_post, X_hatebase, X_profane, X_letters_w2v_features])
        # Double Matrix
        # slow!
        X_previous = matrix_expantion_double(X)
        X = sparse.hstack([X_previous, X]).tocsr()
        
        proba = pretrained_model.predict_proba(X)[:,1]
        proba_list = proba.tolist()
        
        maximum_1.append(max(proba))
        
        num_comment = sum(i > 0.3 for i in proba.tolist())
        num_comments_2.append(sum(i > 0.3 for i in proba.tolist()))
        
        if len(proba_list) == 1:
            max_slope = 0
        else:   
            max_slope = max([x - z for x, z in zip(proba.tolist()[:-1], proba.tolist()[1:])])
            
        max_slope_3.append(max_slope)    
        max_difference_4.append(max(proba_list) - min(proba_list))
        ratio_5.append(num_comment/len(proba_list))
        
#     result[:,0] = np.asarray(maximum_1)

    result[:,0] = np.asarray(num_comments_2)
    result[:,1] = np.asarray(max_slope_3)
    result[:,2] = np.asarray(max_difference_4)
    result[:,3] = np.asarray(ratio_5)
    
    header_trends = ["trends_1", "trends_2", "trends_3", "trends_4"]
    
    return result, header_trends


def matrix_expantion_double(X):
    
    row, col = X.shape
    
    start = 0
    
    new_X = np.zeros(shape=(row, col))
                
    current_post_X = X.todense()
    cumulative_vector = np.zeros(shape=(1,col))
        
    for i in range(row):
                        
        if i == 0 :
            new_X[i, :] = cumulative_vector
        else:
            new_X[i, :] = cumulative_vector/i
                
        cumulative_vector += current_post_X[i]
    
    return new_X 
