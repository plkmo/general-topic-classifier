import os
import re
import math
import spacy
import random
import pandas as pd
from tqdm import tqdm

LABELS_MAP = {('sports', 'athlete', 'fitness'): 'sports', 
            ('business', 'business & finance', 'company'): 'business & finance',
            ('world', 'politics & government', 'officeholder'): 'politics & government',
            ('sci/tech', 'science & mathematics'): 'science & mathematics',
            ('animal', 'pets_animals'): 'pets & animals',
            ('computers & internet', 'tech'): 'computers & internet',
            ('food',): 'food',
            ('health',): 'health',
            ('attraction', 'naturalplace', 'building'): 'travel',
            ('society & culture', 'artist', 'writtenwork', 'fashion', 'literature'): 'society & culture',
            ('education & reference', 'educationalinstitution'): 'education & reference',
            ('entertainment & music', 'album', 'film', 'music', 'celebrities', 'movie', 'games'): 'entertainment & music',
            ('family & relationships',): 'family & relationships',
            ('chitchat', 'joke', 'weather'): 'mundane',
            ('religion'): 'religion',
            ('plant',): 'others'
            }

SUB_REDDITS_MAP = {
    'AskDocs': {'label': 'health', 'sample': 70000},
    'CaregiverSupport': {'label': 'health', 'sample': 70000},
    'healthcare': {'label': 'health', 'sample': 125000},
    'medical': {'label': 'health', 'sample': 125000},
    'medical_advice': {'label': 'health', 'sample': 125000},
    'mentalhealth': {'label': 'health', 'sample': 155000},
    'stopsmoking': {'label': 'health', 'sample': 52000},
    'food': {'label': 'food', 'sample': 500000},
    'Cooking': {'label': 'food', 'sample': 500000},
    'cooking': {'label': 'food', 'sample': 500000},
    'SingaporeEats': {'label': 'food', 'sample': 500000},
    'pets': {'label': 'pets & animals', 'sample': 120000},
    'AnimalsBeingBros': {'label': 'pets & animals', 'sample': 120000},
    'family': {'label': 'family & relationships', 'sample': 115000},
    'Parenting': {'label': 'family & relationships', 'sample': 110000},
    'travel': {'label': 'travel', 'sample': 129000},
    'backpacking': {'label': 'travel', 'sample': 122000},
    'TravelHacks': {'label': 'travel', 'sample': 122000},
    'tech': {'label': 'computers & internet', 'sample': 117000},
    'techology': {'label': 'computers & internet', 'sample': 153000},
    'sports': {'label': 'sports', 'sample': 100000},
    'exercise': {'label': 'sports', 'sample': 100000},
    'fitness': {'label': 'sports', 'sample': 100000},
    'books': {'label': 'education & reference', 'sample': 90000},
    'weather': {'label': 'mundane', 'sample': 105000},
    'Christianity': {'label': 'religion', 'sample': 100000},
    'Christian': {'label': 'religion', 'sample': 70000},
    'Buddism': {'label': 'religion', 'sample': 100000},
    'atheism': {'label': 'religion', 'sample': 100000},
    'Buddhism': {'label': 'religion', 'sample': 100000},
    'Muslim': {'label': 'religion', 'sample': 100000},
    'islam': {'label': 'religion', 'sample': 100000},
    'religion': {'label': 'religion', 'sample': 100000},
    'hindiusm': {'label': 'religion', 'sample': 100000},
    'taoism': {'label': 'religion', 'sample': 100000},
    'jokes': {'label': 'mundane', 'sample': 105000},
    'politics': {'label': 'politics & government', 'sample': 150000},
    'government': {'label': 'politics & government', 'sample': 130000},
    'movies': {'label': 'entertainment & music', 'sample': 110000},
    'personalfinance': {'label': 'business & finance', 'sample': 300000},
    'CryptoCurrency': {'label': 'business & finance', 'sample': 200000},
    'singaporefi': {'label': 'business & finance', 'sample': 4200},
}

nlp = spacy.load("en_core_web_lg")

def clean_sent(sent):
    pattern = r'#.*?'
    sent = re.sub(rf'{pattern}\s', '', sent) # match middle
    sent = re.sub(rf'{pattern}$', '', sent) # match end

    pattern = r'@.*?'
    sent = re.sub(rf'{pattern}\s', '', sent) # match middle
    sent = re.sub(rf'{pattern}$', '', sent) # match end

    pattern = r'http[s]{0,1}://.*?'
    sent = re.sub(rf'{pattern}\s', '', sent) # match middle
    sent = re.sub(rf'{pattern}$', '', sent) # match end

    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    sent = regrex_pattern.sub('', sent)
    sent = sent.encode('ascii', errors='ignore').decode('ascii').strip()

    sent = re.sub(r"( *['’])", "'", sent) # remove extra spaces from ' s          
    sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
    sent = re.sub(r"[\*\"\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#\\]", " ", sent)
    sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
    sent = re.sub('\n{2,}', '\n', sent) # remove extra \n > 1
    sent = re.sub("^ +", "", sent) # remove space in front
    sent = re.sub(r"([\?,!\.]){1,}", r"\1", sent) # remove multiple puncs
    return sent.strip().strip('\n')

def preprocess(ag_news_folder='./data/topic_classification/ag_news_csv',
                yahoo_answers_folder='./data/topic_classification/yahoo_answers_csv',
                dbpedia_folder='./data/topic_classification/dbpedia_csv',
                concet_folder='./data/topic_classification/ConCETData',
                reddit_folder='./data/topic_classification/reddit/export_csvs',
                sg_politics_folder='./data/topic_classification/sg_politics',
                out_folder='./data/topic_classification/processed/'):
    # harmonize topic labels
    labels_map_ = {}
    idx2labels = {}
    for idx, (k, v) in enumerate(LABELS_MAP.items()):
        for key in k:
            labels_map_[key] = v
        idx2labels[idx] = v
    labels2idx = {v:k for k, v in idx2labels.items()}
    labels_map = labels_map_; del labels_map_
    print("LABELS MAPPING:\n", labels_map)

    print("SG Politics data...")
    idx = 0
    for _, file in enumerate(os.listdir(sg_politics_folder)):
        if '.csv' in file:
            if idx == 0:
                df_sg_politics = pd.read_csv(os.path.join(sg_politics_folder, file))
                idx += 1
            else:
                df_sg_politics = df_sg_politics.append(pd.read_csv(os.path.join(sg_politics_folder, file)), \
                                    ignore_index=True, verify_integrity=False, sort=False)

    df_sg_politics.drop_duplicates(subset=['title'], keep='first', inplace=True, ignore_index=True)
    sg_politics_utts, sg_politics_labels = [], []
    for _, row in tqdm(df_sg_politics.iterrows(), total=len(df_sg_politics)):
        title = clean_sent(row['title'])
        if (len(title) > 7) and (len(title) < 200):
            sg_politics_utts.append(title)
            sg_politics_labels.append('politics & government')
        doc = nlp(clean_sent(row['text']))
        sent_ = ""
        for sent in doc.sents:
            if (len(sent.text) > 7) and (len(sent.text) < 200):
                if len(sent_ + ' ' + sent.text) < 200:
                    sent_ += ' ' + sent.text
                    sent_.strip()
                else:
                    sg_politics_utts.append(sent_)
                    sg_politics_labels.append('politics & government')
                    sent_ = sent.text

    del df_sg_politics
    print("Processed SG Politics data rows: ", len(sg_politics_utts))
    print("Sample SG Politics data, labels: ", sg_politics_utts[:3], '\n', sg_politics_labels[:3])
    print("")
    
    print("EXTRACTING AG DATA...")
    with open(os.path.join(ag_news_folder, 'classes.txt'), 'r', encoding='utf8') as f:
        ag_labels_ = [a.strip().strip('\n') for a in f.read().split('\n') if len(a) > 1]
    print("AG labels:\n", ag_labels_)

    ag_utts, ag_labels = [], []
    df_ag = pd.read_csv(os.path.join(ag_news_folder, 'train.csv'), names=['label', 'u1', 'u2'])
    df_ag = df_ag.append(pd.read_csv(os.path.join(ag_news_folder, 'test.csv'), names=['label', 'u1', 'u2']),\
                            ignore_index=True, verify_integrity=False, sort=False)
    
    for idx, row in tqdm(df_ag.iterrows(), total=len(df_ag)):
        ag_label = ag_labels_[int(str(row['label']).strip()) - 1]
        ag_utt = row['u1'].strip().strip('\n') + ' ' + row['u2'].strip().strip('\n')

        if (len(ag_utt) > 7) and (len(ag_utt) < 200):
            ag_labels.append(labels_map[ag_label.lower()])
            ag_utts.append(ag_utt)
    
    del df_ag
    print("Processed AG data rows: ", len(ag_utts))
    print("Sample AG data, labels: ", ag_utts[:3], '\n', ag_labels[:3])
    print("")

    print("EXTRACTING YAHOO ANSWERS DATA...")
    with open(os.path.join(yahoo_answers_folder, 'classes.txt'), 'r', encoding='utf8') as f:
        yh_labels_ = [a.strip().strip('\n') for a in f.read().split('\n') if len(a) > 1]
    print("Yahoo Answers labels:\n", yh_labels_)

    yh_utts, yh_labels = [], []
    df_yh = pd.read_csv(os.path.join(yahoo_answers_folder, 'train.csv'), names=['label', 'u1', 'u2', 'u3'])
    df_yh = df_yh.append(pd.read_csv(os.path.join(yahoo_answers_folder, 'test.csv'), names=['label', 'u1', 'u2', 'u3']),\
                            ignore_index=True, verify_integrity=False, sort=False)
    
    for idx, row in tqdm(df_yh.iterrows(), total=len(df_yh)):
        yh_label = yh_labels_[int(str(row['label']).strip()) - 1]
        yh_utt = row['u1'].strip().strip('\n')

        if (len(yh_utt) > 7) and (len(yh_utt) < 200):
            if str(row['u2']).lower().strip() not in ['', ' ', 'nan', 'none', 'nil']:
                yh_utt += ' ' + row['u2'].strip().strip('\n')
            if str(row['u3']).lower().strip() not in ['', ' ', 'nan', 'none', 'nil']:
                yh_utt += ' ' + row['u3'].strip().strip('\n')

        if (len(yh_utt) < 200):
            yh_labels.append(labels_map[yh_label.lower()])
            yh_utts.append(yh_utt)
    
    del df_yh
    print("Processed Yahoo answers data rows: ", len(yh_utts))
    print("Sample Yahoo answers data, labels: ", yh_utts[:3], '\n', yh_labels[:3])
    print("")

    print("EXTRACTING DBPEDIA DATA...")
    with open(os.path.join(dbpedia_folder, 'classes.txt'), 'r', encoding='utf8') as f:
        db_labels_ = [a.strip().strip('\n') for a in f.read().split('\n') if len(a) > 1]
    print("DBPedia labels:\n", db_labels_)

    db_utts, db_labels = [], []
    df_db = pd.read_csv(os.path.join(dbpedia_folder, 'train.csv'), names=['label', 'u1', 'u2'])
    df_db = df_db.append(pd.read_csv(os.path.join(dbpedia_folder, 'test.csv'), names=['label', 'u1', 'u2']),\
                            ignore_index=True, verify_integrity=False, sort=False)
    
    for idx, row in tqdm(df_db.iterrows(), total=len(df_db)):
        db_label = db_labels_[int(str(row['label']).strip()) - 1]
        db_utt = row['u2'].strip().strip('\n')
        if (len(db_utt) > 7) and (db_label.lower() in labels_map.keys()) and (len(db_utt) < 200):
            db_labels.append(labels_map[db_label.lower()])
            db_utts.append(db_utt)
    
    del df_db
    print("Processed DBPedia data rows: ", len(db_utts))
    print("Sample DBPedia data, labels: ", db_utts[:3], '\n', db_labels[:3])
    print("")

    # ConCET (dialogue_dataset.txt, self_dialogue_dataset_new.txt)
    with open(os.path.join(concet_folder, 'datasets/Final_datasets/dialogue_dataset.txt'), 'r', encoding='utf8') as f:
        concet_data = f.read().split('\n')

    with open(os.path.join(concet_folder, 'datasets/Final_datasets/self_dialogue_dataset_new.txt'), 'r', encoding='utf8') as f:
        concet_data += f.read().split('\n')
    
    concet_data = [a.strip().strip("\n") for a in concet_data if (len(a.strip().strip("\n")) > 3)]
    concet_data = list(set(concet_data))
    print("ConCET data rows: %s" % len(concet_data))
    print(concet_data[:10])
    print("Extracting ConCET data...")
    concet_utts, concet_labels = [], []
    for row in tqdm(concet_data):
        try:
            label, utt = row.split('\t')
            label = re.sub('__label__', '', label).strip()
            utt_ = utt.split(' ')
            
            if len(utt_) < 3: # ignore super short utterances
                continue

            if len(utt_) > 200: # ignore if too long
                continue

        except Exception as e:
            print("ERROR: ", e)
            print("OFFENDING ROW: ", row)
            continue
        
        if label.lower() in labels_map.keys():
            concet_utts.append(utt)
            concet_labels.append(labels_map[label.lower()])
    
    print(concet_utts[:5])
    print(concet_labels[:5])
    print("Unique labels: ", set(concet_labels))
    print("Processed ConCET data rows: ", len(concet_labels))
    print("Sample ConCET data, labels: ", concet_utts[:3], '\n', concet_labels[:3])
    print("")

    # Sub-reddits
    print("EXTRACTING REDDIT DATA...")
    reddit_utts, reddit_labels = [], []
    for subreddit_folder in tqdm(os.listdir(reddit_folder)):
        if subreddit_folder in SUB_REDDITS_MAP.keys():
            subreddit_label = SUB_REDDITS_MAP[subreddit_folder]['label']
            subreddit_utts = []
            for file in os.listdir(os.path.join(reddit_folder, subreddit_folder)):
                if '.csv' in file:
                    print("File: %s" % file)
                    df = pd.read_csv(os.path.join(os.path.join(reddit_folder, subreddit_folder), file))
                    for _, row in df.iterrows():
                        convo = str(row['conversation'])

                        # too short
                        if (convo.lower().strip() in ['none', 'nil', 'na', '', ' ', 'nan']) or (len(convo.strip().strip('\n')) < 15):
                            continue

                        convo_check = re.split(r'[\d+]\)', re.sub('(\(.+\))', '', convo))
                        convo_check = [c for c in [clean_sent(e) for e in convo_check if (len(e) > 0)] if ((len(c) > 0) and (len(c) < 230))]
                        subreddit_utts.append(" ".join(convo_check[:2]).strip().strip('\n'))
            
            # sampling
            if len(subreddit_utts) > 0:
                reddit_utts.extend(random.sample(subreddit_utts, min(SUB_REDDITS_MAP[subreddit_folder]['sample'], len(subreddit_utts)) ))
                ls = [subreddit_label for _ in range(min(SUB_REDDITS_MAP[subreddit_folder]['sample'], len(subreddit_utts)))]
                reddit_labels.extend(ls)
                print("Added %d rows for label %s" % (len(ls), subreddit_label))

    print("Total data rows from reddit: %d" % (len(reddit_utts)))
                
    print("Combining datasets...")
    combined_utts, combined_labels = [], []
    combined_utts += ag_utts + yh_utts + db_utts + concet_utts + reddit_utts + sg_politics_utts
    combined_labels += ag_labels + yh_labels + db_labels + concet_labels + reddit_labels + sg_politics_labels
    del ag_utts, ag_labels, yh_utts, yh_labels, db_utts, db_labels, concet_utts, concet_labels, \
        reddit_utts, reddit_labels, sg_politics_utts, sg_politics_labels
    
    # Stats
    print("Total data rows: ", len(combined_utts))
    df_data = pd.DataFrame(data={'utterance': combined_utts, 'label': combined_labels})
    df_data['tokens_length'] = df_data['utterance'].apply(lambda x: len(x.split(' ')))
    print("Labels distribution:\n", df_data["label"].value_counts(), '\n')
    print("Average tokens length per label:\n", df_data[["label", "tokens_length"]].groupby('label').mean(), '\n')
    print("Samples per label:\n", df_data[["label", "utterance"]].groupby('label').sample(n=3), '\n')
    del df_data

    print("Saving...")
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
        print("Created directory ", out_folder)

    # 1M rows per file
    for i in range(math.ceil(len(combined_utts)/1000000)):
        filename = 'tc_data_%d.csv' % (i + 1)
        df = pd.DataFrame(data={'text': combined_utts[i*1000000:((i + 1)*1000000)], 'label': combined_labels[i*1000000:((i + 1)*1000000)]})
        df.to_csv(os.path.join(out_folder, filename))
        print("Saved to %s" % (os.path.join(out_folder, filename)))
    return

if __name__ == '__main__':
    preprocess()