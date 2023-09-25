# General Topic Classifier for Short-to-Medium Length Text
_One_ _model_, _16_ _classes_, _3M+_ _rows_ _of_ _labelled_ _data_   

The infererence-optimised onnx model "onnx_optimised_mpnet_trained" to be deployed here, and is created from training/   
The optimised model runs reasonably fast on CPU as well.   
See training/README.md for details on dataset, training & benchmarks.

The model deployed here is based on MPNet (https://arxiv.org/abs/2004.09297), and has been fine-tuned on 3M+ data rows. It classifies short-to-medium length text into one of the 16 general topics: 
- entertainment & music  
- health  
- business & finance  
- sports  
- religion  
- pets & animals  
- family & relationships  
- food  
- travel  
- education & reference  
- politics & government  
- society & culture   
- mundane  
- computers & internet  
- science & mathematics  
- others   

If you find this useful, please consider sponsoring by clicking the sponsor button at the top.   
For more details, check out this published [article](https://medium.com/towards-data-science/bert-s-for-relation-extraction-in-nlp-2c7c3ab487c4). 

## Deploying the model
Download model artefacts [here](https://plkmo-general-topic-classifier-model.s3.amazonaws.com/onnx/general_topic_classifier.mar) and save as onnx/general_topic_classifier.mar
```bash
# build image
export DOCKER_BUILDKIT=1
docker build -f Dockerfile_cpu_multi -t plkmo/general_topic_classifier:cpu . # CPU
docker build -f Dockerfile_cuda -t plkmo/general_topic_classifier:cuda . # gpu

# create torchserve mar file from trained model (only needed if you've trained the model)
# (skip this if onnx/general_topic_classifier.mar already exists 
# or downloaded from above link)
torch-model-archiver --model-name general_topic_classifier --version 1.0 \
                        --serialized-file onnx/onnx_optimised_mpnet_trained \
                        --handler ./torchserve_handler.py \
                        --extra-files "./setup_config.json,./training/MPNetTokenizer/special_tokens_map.json,./training/MPNetTokenizer/tokenizer_config.json,./training/MPNetTokenizer/vocab.txt"
# This generates 'general_topic_classifier.mar' at the project directory. Move this file to 'onnx/general_topic_classifier.mar'

# path to onnx folder (change this to your path)
export ONNX_PATH=/Users/weeteesoh/Desktop/general-topic-classifier/onnx
docker run -d --name general_topic_classifier \
    --restart always \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --log-opt max-size=10m \
    --log-opt max-file=1 \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    -p 7070:7070 \
    -p 7071:7071 \
    -v $ONNX_PATH:/onnx \
    plkmo/general_topic_classifier:cpu

```

## Testing the inference endpoint    
```bash
curl --location --request POST 'http://localhost:8080/predictions/general_topic_classifier' \
--header 'Content-Type: application/json' \
--data-raw '{
    "sents": [
        "Happy hours. Now I want a pet porcupine",
        "Cat saves the day Hahaha,very clever!",
        "Maldives shuts down all spas Would be easier to swallow, no pun, if one replaced islamist for feminist?",
        "Reports about many smaller confrontations, like one at a womens college in Bengaluru, spread via social media.",
        "The protests over Indias Citizenship Amendment Act",
        "Its not just the real time experience from other countries that informs policy responses."
    ]
}'

# response (list of [confidence, label])
[
  [
    "0.99997365",
    "pets & animals"
  ],
  [
    "0.99981374",
    "pets & animals"
  ],
  [
    "0.81139916",
    "travel"
  ],
  [
    "0.99999857",
    "politics & government"
  ],
  [
    "0.9998914",
    "politics & government"
  ],
  [
    "0.9999852",
    "politics & government"
  ]
]

curl --location --request POST 'http://localhost:8080/predictions/general_topic_classifier' \
--header 'Content-Type: application/json' \
--data-raw '{
    "sents": [
        "You should invest in BitCoin, its gonna rise alot!",
        "If you buy Sembcorp marine stocks, you would have won alot if you held.",
        "mortgage interest rates should be fine for investors who bought property late last year. I believe the rates will be fixed for next 3 years if they bought late last year and the rate should be much better than now...",
        "There are some risk considerations that we should be aware of when we are spending down our assets. The majority of the folks are at a stage where they focus on wealth accumulation, but, what I realize is that the considerations when you reach the stage of withdrawing part of your wealth from your wealth assets are very different. ",
        "Asian Shares Mostly Lower On Recession Fears Asian stocks ended mostly lower on Tuesday after Wall Street entered the bear market on fears that aggressive rate tightening by the Federal Reserve would push the world largest economy into recession",
        "HSBC launches global programme with IMF and WHO for financial literacy and empowerment"
    ]
}'

# response (list of [confidence, label])
[
  [
    "0.9999747",
    "business & finance"
  ],
  [
    "0.99952066",
    "business & finance"
  ],
  [
    "0.99934846",
    "business & finance"
  ],
  [
    "0.99907744",
    "business & finance"
  ],
  [
    "0.98783314",
    "business & finance"
  ],
  [
    "0.73752135",
    "business & finance"
  ]
]
```

## License  
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg