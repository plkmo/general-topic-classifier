import os
import re
import string
import numpy as np
from transformers import MPNetTokenizer
import torch
import onnxruntime as rt

EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class GenTopicClassifier(object):
    """
    General Topic Classifier handler class. This handler takes a string
    as input and returns the profanity label
    """

    def __init__(self):
        super(GenTopicClassifier, self).__init__()
        self.initialized = False
        self.ascii = set(string.ascii_letters + string.ascii_uppercase + string.ascii_lowercase)
        self.labels_dict = {'l2idx': {'entertainment & music': 0, 'sports': 1, 'travel': 2, \
                            'computers & internet': 3, 'mundane': 4, 'society & culture': 5,
                            'food': 6, 'pets & animals': 7, 'health': 8,
                            'family & relationships': 9, 'religion': 10,
                            'education & reference': 11, 'politics & government': 12,
                            'business & finance': 13, 'science & mathematics': 14, 'others': 15}}

        self.labels_dict['idx2l'] = {v:k for k,v in self.labels_dict['l2idx'].items()}
        self.onnx_runtime = str(rt.get_device())
        print("ONNX Runtime: %s" % self.onnx_runtime)
    
    def initialize(self, context):
        print('Initialise model')
        properties = context.system_properties
        print(properties)
        model_dir = properties.get("model_dir")
        
        if not model_dir.endswith('/'):
            model_dir += '/'

        self.tokenizer = MPNetTokenizer.from_pretrained(model_dir)
        self.ort_session = rt.InferenceSession(os.path.join(model_dir, 'onnx_optimised_mpnet_trained'),\
                                            providers=EP_list)
        self.initialized = True
    
    def clean_sent(self, sent):
        sent = ''.join([char for char in sent if char in self.ascii])
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

    def preprocess(self, data):
        inputs = data[0].get("data")
        if inputs is None:
            inputs = data[0].get("body")

        #inputs['sents'] = [self.clean_sent(sent) for sent in inputs['sents']]
        inputs = self.tokenizer(inputs['sents'], padding=True, truncation=True, return_tensors="pt" ,\
                                max_length=500)
        return inputs

    def inference(self, inputs):
        ort_inputs = {'input_ids': to_numpy(inputs['input_ids']) if (self.onnx_runtime.lower() == 'cpu') else inputs['input_ids'],\
                    'attention_mask': to_numpy(inputs['attention_mask']) if (self.onnx_runtime.lower() == 'cpu') else inputs['attention_mask']}
        ort_outs = self.ort_session.run(None, ort_inputs)
        confidence = torch.softmax(torch.FloatTensor(ort_outs[0]) if not isinstance(ort_outs[0], torch.FloatTensor) else ort_outs[0], 1)
        confidence, predicted = torch.max(confidence, 1)
        predicted = list(predicted.cpu().numpy()) if not (self.onnx_runtime.lower() == 'cpu') else list(predicted.numpy())
        confidence = list(confidence.cpu().numpy()) if not (self.onnx_runtime.lower() == 'cpu') else list(confidence.numpy())
        return confidence, predicted

    def postprocess(self, confidence, predicted):
        output = [[str(c), self.labels_dict['idx2l'][p]] for c, p in zip(confidence, predicted)]
        return [output]

_service = GenTopicClassifier()

def handle(data, context):
    """
    Entry point for GenTopicClassifier handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data_ = _service.preprocess(data)
        confidence, predicted = _service.inference(data_)
        data_ = _service.postprocess(confidence, predicted)

        return data_
    except Exception as e:
        raise Exception("Unable to process input data. " + str(e))