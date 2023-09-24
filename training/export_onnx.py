from argparse import ArgumentParser
import time
import torch
from transformers import MPNetTokenizer, MPNetForSequenceClassification

def convert_to_onnx():
    tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')
    print("Saving tokenizer...")
    tokenizer.save_pretrained("./MPNetTokenizer/")
    model = MPNetForSequenceClassification.from_pretrained('output/mpnet_topicclassifier/best_model/model', num_labels=16)

    labels_dict = {'l2idx': {'entertainment & music': 0, 'sports': 1, 'travel': 2, \
                                'computers & internet': 3, 'mundane': 4, 'society & culture': 5,
                                'food': 6, 'pets & animals': 7, 'health': 8,
                                'family & relationships': 9, 'religion': 10,
                                'education & reference': 11, 'politics & government': 12,
                                'business & finance': 13, 'science & mathematics': 14, 'others': 15}}

    labels_dict['idx2l'] = {v:k for k,v in labels_dict['l2idx'].items()}

    # "Hello, my dog is cute"
    #inputs = {'input_ids': torch.LongTensor([[    0,  7596,  1014,  2030,  3903,  2007, 10144,     2]]), \
    #        'attention_mask': torch.FloatTensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
    inputs = tokenizer(["Hello, my dog is cute", "Hello, my cat is really quite cute."],\
                    padding=True, truncation=True, return_tensors="pt")
    output = model(**inputs)
    output_ids = torch.softmax(output.logits, dim=1).max(dim=1).indices

    with torch.no_grad():
        print("Converting to onnx...")
        torch.onnx.export(model,\
                    args=(inputs['input_ids'], inputs['attention_mask']),
                    f='onnx_mpnet_trained',
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=["input_ids", "attention_mask"],
                    output_names=['logits'],
                    dynamic_axes={
                        "input_ids": {0: 'batch_size', 1: "sequence"},
                        "attention_mask": {0: 'batch_size', 1: "sequence"},
                        "logits": {0: "batch_size"}
                    },
                    verbose=True)
    print("Done and saved model as onnx_mpnet_trained")
    return

def optimize_onnx_model():
    print("Optimising...")
    from onnxruntime.transformers import optimizer
    optimized_model = optimizer.optimize_model("onnx_mpnet_trained", model_type='bert',\
                                            num_heads=12, hidden_size=768,)
    optimized_model.save_model_to_file("onnx_optimised_mpnet_trained")
    print("Done and saved model as onnx_optimised_mpnet_trained")
    return

def check_onnx_model():
    import numpy as np
    import onnx
    import onnxruntime
    model_name = 'onnx_optimised_mpnet_trained'
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx.checker.check_model(model_name)

    # pytorch inference
    tokenizer = MPNetTokenizer.from_pretrained("./MPNetTokenizer/")
    model = MPNetForSequenceClassification.from_pretrained('output/mpnet_topicclassifier/best_model/model', num_labels=16)

    labels_dict = {'l2idx': {'entertainment & music': 0, 'sports': 1, 'travel': 2, \
                                'computers & internet': 3, 'mundane': 4, 'society & culture': 5,
                                'food': 6, 'pets & animals': 7, 'health': 8,
                                'family & relationships': 9, 'religion': 10,
                                'education & reference': 11, 'politics & government': 12,
                                'business & finance': 13, 'science & mathematics': 14, 'others': 15}}

    labels_dict['idx2l'] = {v:k for k,v in labels_dict['l2idx'].items()}

    # "Hello, my dog is cute"
    start = time.time()
    #inputs = {'input_ids': torch.LongTensor([[    0,  7596,  1014,  2030,  3903,  2007, 10144,     2]]), \
    #        'attention_mask': torch.FloatTensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
    inputs = tokenizer(["I went to Korea as a tourist.", "I was in Singapore for a tour"],\
                padding=True, truncation=True, return_tensors="pt")
    output = model(**inputs)
    output_ids = torch.softmax(output.logits, dim=1).max(dim=1).indices
    print("Pytorch inference took %.3f s" % (time.time() - start))

    print("Starting onnx runtime...")
    ort_session = onnxruntime.InferenceSession(model_name, providers=EP_list)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # compute ONNX Runtime output prediction
    ort_inputs = {'input_ids': to_numpy(inputs['input_ids']),\
                    'attention_mask': to_numpy(inputs['attention_mask'])}
    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    confidence = torch.softmax(torch.FloatTensor(ort_outs[0]) if not isinstance(ort_outs[0], torch.FloatTensor) else ort_outs[0], 1)
    confidence, predicted = torch.max(confidence, 1)
    print("Confidence, pred : %s, %s" % (confidence, str(predicted)))
    # compare ONNX Runtime and PyTorch results
    print("PYTORCH OUTPUTS: ", to_numpy(output['logits']))
    print("ONNX OUTPUTS: ", ort_outs[0])
    print("ONNX inference took %.3f s" % (time.time() - start))
    np.testing.assert_allclose(to_numpy(output['logits']), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    # General Parameters
    parser.add_argument('--action', type=str, default='convert', help='convert, optimize, check')
    args = parser.parse_args()

    if args.action == 'convert':
        convert_to_onnx()
    elif args.action == 'optimize':
        optimize_onnx_model()
    elif args.action == 'check':
        check_onnx_model()