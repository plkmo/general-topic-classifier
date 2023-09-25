# Training a Classifier for General Topics in Short-to-Medium Length Text

## Requirements
```bash
python3 -m pip install -r requirements.txt
```
If you find this useful, please consider sponsoring by clicking the sponsor button at the top.   

## Data Preparation
Data sources:   
- AG News  
- Yahoo Answers  
- DBPedia  
- ConCET self-dialogue data  
- selected subreddits (crawled using timesearch repo - https://github.com/voussoir/timesearch)  

Contact me if you need the full dataset. Once obtained, save it as './data' folder here.      
After the raw dataset files are obtained above, run the below command to compile the datasets.   
```bash
python3 combine_datasets.py
```
This saves the final processed & compiled training data to the directory './data/topic_classification/processed/', which is used for training. 


## Data statistics
Note that only a small subset of crawled reddit data is used for training. For the specific subreddit distribution of the dataset used, see SUB_REDDITS_MAP in combine_datasets.py   

```bash
Total data rows from reddit: 1737126
Combining datasets...
Total data rows from all sources:  3013399
Labels distribution:
 entertainment & music     612742
health                    381408
business & finance        329025
sports                    315968
religion                  255347
pets & animals            243162
family & relationships    167804
food                      142673
travel                    114434
education & reference     102501
politics & government      90535
society & culture          66347
mundane                    65277
computers & internet       59131
science & mathematics      45813
others                     21232
Name: label, dtype: int64 

Average tokens length per label:
                         tokens_length
label                                
business & finance          25.461503
computers & internet        17.090832
education & reference       25.338572
entertainment & music       13.799702
family & relationships      30.866273
food                        28.267941
health                      29.767784
mundane                     11.495381
others                      17.963122
pets & animals              23.996336
politics & government       24.107693
religion                    28.574837
science & mathematics       22.518586
society & culture           21.176692
sports                      20.929091
travel                      25.768006 

Samples per label:
                           label                                          utterance
147868       business & finance  Fill in the blank: the rate of marching at 60 ...
1782124      business & finance  Any active defi yield farmers here? Would love...
2533335      business & finance  Are Whales too much a religion in crypto? If y...
416267     computers & internet  how many 0's in a gig? 1gb=1024 mb\n1mb=1024 k...
1109454    computers & internet       just let me know about alliance data systems
116575     computers & internet  How do you erase entries from the "Search Web"...
474406    education & reference  St Thomas More College (STMC) is a Catholic co...
1681025   education & reference  She was only five and a half years old, but sh...
1657236   education & reference  If you live in New Zealand and if you like sci...
1145002   entertainment & music           play a alternative rap genre from eminem
592590    entertainment & music  Everybody's Got a Story is Amanda Marshall's t...
2474629   entertainment & music  I was just thinking of the movie Night Shift t...
1908905  family & relationships  Am I forcing my kids to grow up too fast? I do...
1949667  family & relationships  Girlfriend's daughter doesn't know what to cal...
1928472  family & relationships  Great News Regarding Daughter That is amazing ...
1362966                    food  Let's talk Sides. Post your best accompaniment...
1322712                    food  The Science of Pan Frying. Some really good vi...
2357404                    food  Bacon of the Sea Smoked Salmon Belly King strips.
343478                   health              What is this? what is cervical fluid?
1289299                  health  Can I write a summary of my symptoms on a pape...
1433265                  health  it has been 4 days since i quit smoking and i ...
2892308                 mundane  0 to 200 in 6 seconds Here in India, the joke ...
1069933                 mundane                  who the hell are all these people
2884817                 mundane  Quick thinking So were the girls eaten by the ...
585933                   others  Polystachya maculata is a species of orchid en...
588407                   others  Shorea tumbuggaia is a species of plant in the...
577877                   others  Hakea aculeata commonly known as the Column Ha...
547008           pets & animals  Hemonia murina is a moth of the Arctiidae fami...
2758104          pets & animals  My 10 month old pup Finn tore a copperhead in ...
546709           pets & animals  Volvarina pseudophilippinarum is a species of ...
1796080   politics & government  President Clinton The United States on Track t...
2994612   politics & government  \nXi Jinping Thought was incorporated into the...
3001524   politics & government  I hope the Land Transport Authority could scra...
2490307                religion  Nowadays people are saying almost everything i...
1883334                religion  Do you think this is true? Why or why not Go a...
1816256                religion  Can you be a homosexual and still be close to ...
69102     science & mathematics  satellite of a planet? A satellite is the moon...
34973     science & mathematics  how is the climate in chennai city? right now ...
182935    science & mathematics  What is the size of the worlds largest penis? ...
486541        society & culture  Joseph Bentwich (Hebrew: 'יוסף בנטוויץ‎; born ...
261872        society & culture  The wise man built his house upon the??????? T...
187353        society & culture  What do women wear to a Bar Mitzvah? Depends o...
1730680                  sports  Who thought that this suit was the best idea? ...
1726372                  sports  Ballboy tells China goalkeeper which way to di...
1696167                  sports  Anybody else notice the frequency of falling i...
1560971                  travel  Wind Rivers Wyoming did not disappoint. 25 mil...
1577984                  travel  Will Wild's Outdoor Adventures Hiking Along A ...
544832                   travel  The Somonița River ( Hungarian: Szomonyica-pat...
```


### Training script ###
Supported Models:
1. conv-bert-base     
2. distilbert-base-uncased   
3. bert-base-uncased      
4. microsoft/mpnet-base   

To train, run the script `model_training.py`  

```bash
# Using convbert
python3 model_training.py --model_selection convbert \
                            --job_id convbert \
                            --pad True --batch_size 32 \
                            --num_epoch 10 --es_patience 3 \
                            --gradient_accumulation_steps 1

python3 model_training.py --model_selection convbert \
                        --job_id convbert \
                        --pad True --batch_size 256 \
                        --num_epoch 12 --es_patience 3 \
                        --gradient_accumulation_steps 1 \
                        --use_cuda True

# Using mpnet (best results)
python3 model_training.py --model_selection mpnet \
                        --job_id mpnet \
                        --pad True --batch_size 96 \
                        --num_epoch 13 --es_patience 3 \
                        --gradient_accumulation_steps 2 \
                        --use_cuda True
```

## Benchmark results
```bash
# Best model: MPNet
{
      "epoch": 15.0,
      "eval_accuracy": 0.9358124937730398,
      "eval_business_n_finance_f1": 0.9251001746635158,
      "eval_business_n_finance_precision": 0.9375260308204915,
      "eval_business_n_finance_recall": 0.9129993916041371,
      "eval_f1": 0.9180528067593838,
      "eval_loss": 0.19770504534244537,
      "eval_politics_f1": 0.9079620160701243,
      "eval_politics_precision": 0.9000724112961622,
      "eval_politics_recall": 0.9159911569638909,
      "eval_precision": 0.9180914902905605,
      "eval_recall": 0.9183993059897613,
      "eval_religion_f1": 0.9714586347494409,
      "eval_religion_precision": 0.9744063324538259,
      "eval_religion_recall": 0.96852871754524,
      "eval_runtime": 103.4828,
      "eval_samples_per_second": 387.968,
      "eval_steps_per_second": 4.049,
      "step": 202830
    }
```

## Packaging for inference
Make sure that the trained model is at ./output folder.    
Eg. ./output/mpnet_topicclassifier should contain the trained model files of mpnet, that is generated from model_training.py script.   

1. Convert the model into onnx framework.     
```bash
python3 export_onnx.py --action convert
```
The converted model will be saved as 'onnx_mpnet_trained'.   

2. Optimize the converted onnx model.   
```bash
python3 export_onnx.py --action optimize
```
The optimized model will be saved as 'onnx_optimised_mpnet_trained'  

3. Check the integrity of the converted model ('onnx_optimised_mpnet_trained').    
```bash
python3 export_onnx.py --action check
```

## License  
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg