# General Topic Classifier Deployment
The onnx model "onnx_optimised_mpnet_trained" to be deployed here, and is created from training/   
See training/README.md for details.   

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

## Deploying the model
```bash
# build image
export DOCKER_BUILDKIT=1
docker build -f Dockerfile_cpu_multi -t plkmo/general_topic_classifier:cpu . # CPU
docker build -f Dockerfile_cuda -t plkmo/general_topic_classifier:cuda . # gpu

# create torchserve mar file (skip this if onnx/general_topic_classifier.mar already exists)
torch-model-archiver --model-name general_topic_classifier --version 1.0 \
                        --serialized-file onnx/onnx_optimised_mpnet_trained \
                        --handler ./torchserve_handler.py \
                        --extra-files "./setup_config.json,./training/MPNetTokenizer/special_tokens_map.json,./training/MPNetTokenizer/tokenizer_config.json,./training/MPNetTokenizer/vocab.txt"
# This generates 'general_topic_classifier.mar' at the project directory. Move this file to 'onnx/general_topic_classifier.mar'

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
    -v ./onnx:/onnx \
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

```


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg