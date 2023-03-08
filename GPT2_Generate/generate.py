import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import re
import argparse
import json
torch.cuda.set_device(0)
model_name = 'gpt2-large'  
model_name_back = 'checkpoint-2673000'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model_back = GPT2LMHeadModel.from_pretrained(model_name_back)
model.config.pad_token_id = model.config.eos_token_id
model_back.config.pad_token_id = model_back.config.eos_token_id
model_back.cuda()
model.cuda()

def load_vocab(filename):
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    eid2idx = {w: i for i, w in enumerate(keywords)}
    return eid2name, keywords, eid2idx


def generate_part2(index, indexed_tokens, num_to_generate, count_0, dict1):
    dict = {}
    context_min_length, context_max_length = 10, 15
    predicted_text = tokenizer.decode(
        indexed_tokens + [index])
    predicted_index = tokenizer.encode(predicted_text)
    length = len(predicted_index)
    tokens_tensor = torch.tensor([predicted_index])
    tokens_tensor = tokens_tensor.cuda()
    outputs = model_back(tokens_tensor)
    predictions = outputs[0]
    probs, indices1 = torch.topk(predictions[0, -1, :], num_to_generate)
    for j in indices1:
        predicted_text = tokenizer.decode(
            predicted_index + [j])
        inputs = tokenizer(
            predicted_text, return_tensors="pt").input_ids
        inputs = inputs.cuda()
        outputs_back = model_back.generate(
            inputs, min_length=context_min_length, max_length=length + context_max_length, num_beams=100, num_return_sequences=1, no_repeat_ngram_size=2)
        outputs = tokenizer.batch_decode(
            outputs_back, skip_special_tokens=True)
        for sentence in outputs:
            count = count_0
            sentence = re.split(
                r'([,.:;?`~!#$%&+={}\"@^*])', sentence)
            sentence = ' '.join(sentence)
            sentence = sentence.split()
            if '\'s' in input_1 or '\'' in input_1:
                count -= 1
            if '\'s' in input_1 and input_1[-1] != '\'s':
                pos = 0
                for iii in range(len(input_1)):
                    if '\'s' == input_1[iii]:
                        pos = iii
                        break
                lian = input_1[pos-1]+(input_1[pos])
                sentence[count-pos] = lian
                sentence[count-1-pos] = input_1[pos+1]
            if '.' in input_1:
                pos = 0
                if ' ' in input:
                    a = input.split()
                    if '.' in a[0] and a[0] != '.':
                        pos = len(a[0].split('.'))
                        sentence[pos-1] = a[0]
                        for i in range(pos-1):
                            sentence[i] = ''
                    elif '.' in a[1]:
                        sentence[1] = a[1]
                        for iii in range(len(a[1])-1):
                            sentence[2+iii] = ''
                else:
                    sentence[count-1] = input
                    for iii in range(count-1):
                        sentence[iii] = ''

            sentence = ' '.join(sentence[::-1])
            inputs = tokenizer(
                sentence, return_tensors="pt").input_ids
            inputs = inputs.cuda()
            outputs_1 = model.generate(
                inputs, min_length=context_min_length + context_max_length, max_length=length + 2 * context_max_length,  no_repeat_ngram_size=2)
            outputs = tokenizer.batch_decode(
                outputs_1, skip_special_tokens=True)
            for j in range(len(outputs)):
                outputs[j] = outputs[j].replace('\n', ' ')
            for sentence_1 in outputs:
                dict1["start"] = 0
                sentence_1 = re.split(
                    r'([,.:;?`~!#$%&+={}\"@^*])', sentence_1)
                sentence_1 = ' '.join(sentence_1)
                sentence_1 = sentence_1.split()
                dict['tokens'] = sentence_1
                for ii in range(len(sentence_1)):
                    if (ii+count-1) >= len(sentence_1):
                        break
                    if '\'s' not in input_1 and input_1[0] in sentence_1[ii] and input_1[-1] in sentence_1[ii+count-1]:
                        dict1["start"] = ii
                        dict1["end"] = ii+count-1
                        break
                    if '\'s' in input_1:
                        if input_1[0] in sentence_1[ii] and input_1[-1] in sentence_1[ii+count-1]:
                            dict1["start"] = ii
                            dict1["end"] = ii+count-1
                if dict1["start"] != 0:
                    dict["entityMentions"] = [dict1]
                    with open("gpt2.json", "a", encoding='utf-8') as f:
                        f.write(json.dumps(dict))
                        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dataset', default='./wiki', help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    args = parser.parse_args([])
    eid2name, keywords, eid2idx = load_vocab(
        os.path.join(args.dataset, args.vocab))
    dict1 = {}
    num_topk = 35
    total_context = 130
    num_index = [2, 4, 6, 7, 16]
    num_context = [15, 10, 5, 2, 1]
    num_generate = [num_context[i] for i in range(len(num_index)) for j in range(i)]
    assert num_topk == sum(num_index)
    assert total_context == sum(num_generate)
    torch.set_grad_enabled(False)
    for m in keywords:
        count_0 = 0
        dict1["entityId"] = m
        input = eid2name[m]
        input_for_count = input[:]
        if '\'s' not in input:
            input_for_count = re.split(
                r'([,.:;?`~!#$%&+={}\"@^*])', input_for_count)
            input_for_count = ' '.join(input_for_count)
            input_for_count = input_for_count.split()
            count_0 = len(input_for_count)
            input = input.split()
            input_1 = input_for_count[:]
        else:
            input = input.split()
            count_0 = len(input)
            input_1 = input[:]
        input = ' '.join(input[::-1])
        indexed_tokens = tokenizer.encode(input)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.cuda()
        outputs = model_back(tokens_tensor)
        predictions = outputs[0]
        probs, indices = torch.topk(predictions[0, -1, :], num_topk)
        for i in range(len(indices)):
            index, num_gen = indices[i], num_generate[i]
            generate_part2(index, indexed_tokens,
                           num_gen, count_0, dict1)