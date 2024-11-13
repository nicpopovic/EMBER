from datasets import load_dataset
from src.data.ner import load_labelmaps
from src.data.loaders import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity
from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
import string
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tner/ontonotes5", help="dataset")
parser.add_argument("--model", type=str, default="gpt2-xl", help="model")
parser.add_argument("--layer", type=int, default=34, help="layer")
parser.add_argument("--k", type=int, default=5, help="k")
parser.add_argument("--n_query", type=int, default=10, help="n_query")
parser.add_argument("--n_episodes", type=int, default=100, help="n_episodes")
args = parser.parse_args()


random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10)) 
print(random_string)


dataset_id = args.dataset
dataset = load_dataset(dataset_id)
model_name = args.model
layer_id = args.layer
k = args.k
n_query = args.n_query
n_episodes = args.n_episodes

model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True, return_dict=True, device_map="auto", cache_dir="hfcache/").half()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir="hfcache/")

dataset_gen = FeatureGenDataset(dataset_id, "validation", dev=False, tokenizer=tokenizer, force_bos=True, rel=False, coref=False)

type_map = load_labelmaps(dataset_id)["ner"]

types_index = [[] for _ in range(len(type_map))]

for i, sample in enumerate([x for x in dataset_gen]):
    for ent in sample['entities_tokenized']:
        types_index[torch.argmax(ent[0][1])].append(i)

types_index = [list(set(x)) for x in types_index]

def sample_episode(num_k, num_queries, dataset, types_index):

    counts = [0] * len(types_index)

    support_indices = []
    for i in range(len(types_index)):
        num_k_for_type = num_k - counts[i]
        if num_k_for_type < 1 or i == 0:
            continue
        support_indices_for_type = random.sample([x for x in types_index[i] if x not in support_indices], num_k)
        support_indices.extend(support_indices_for_type)
        counts[i] += num_k_for_type
        
    query_candidates = [x for x in range(len(dataset)) if x not in support_indices]
    query_indices = random.sample(query_candidates, num_queries)

    return support_indices, query_indices

def get_features_for_sample(sample, llm, layer_id=9, n_classes=5):
    input_ids = torch.tensor(sample['tokens'])
    #print(input_ids.shape)
    outputs = llm(input_ids.to(llm.device).unsqueeze(0), output_attentions=True)
    
    attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()
    attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)

    # create label tensors
    lb_tensor_pos = torch.zeros((attentions.shape[0], attentions.shape[1], attentions.shape[2]))
    lb_tensor_neg = torch.ones((attentions.shape[0], attentions.shape[1], attentions.shape[2]))
    #print(sample['entities_tokenized'])
    for (y, x), label in [x[0] for x in sample['entities_tokenized']]:
        
        span = tokenizer.decode(sample['tokens'][y:x+1])
        span_prev = tokenizer.decode(sample['tokens'][y-1:x+1])
        span_next = tokenizer.decode(sample['tokens'][y:x+2])

        if span[0] not in " '.,;:?!><()":
            if span_prev[:-len(span)][-1] not in " '.,;:?!><":
                #print(f"discarding: '{span}'", span_prev)
                lb_tensor_neg[0,x+1,y] = 0
                continue

        if span[-1] not in " '.,;:?!><()":
            if span_next[len(span):][0] not in " '.,;:?!><":
                #print(f"discarding: '{span}'", span_next)
                lb_tensor_neg[0,x+1,y] = 0
                continue     
            
        label = torch.argmax(label)
        #print(label, positive_index, label == positive_index)
        lb_tensor_pos[0,x+1,y] = 1
        lb_tensor_neg[0,x+1,y] = 0
        #lb_tensor_pos[0,x+1,y] = 1
        #lb_tensor_neg[0,x+1,y] = 0

    causal_mask = (-1 * (torch.triu(torch.ones(input_ids.size(0), input_ids.size(0)), diagonal=1) - 1)).bool().unsqueeze(0)
    causal_mask = torch.logical_and(causal_mask, lb_tensor_neg.bool()).unsqueeze(-1)
    attention_positives = torch.masked_select(attentions, lb_tensor_pos.unsqueeze(-1).bool()).view(-1, attentions.size(-1))
    attention_negatives = torch.masked_select(attentions, causal_mask).view(-1, attentions.size(-1))


    h_s_at_layer = outputs.hidden_states[layer_id].detach().cpu()
    #print(h_s_at_layer.shape)
    # create label tensors
    lb_tensor_pos = [torch.zeros((h_s_at_layer.shape[0], h_s_at_layer.shape[1], 1)) for _ in range(n_classes)]
    lb_tensor_neg = torch.ones((h_s_at_layer.shape[0], h_s_at_layer.shape[1], 1))
    #print(sample['entities_tokenized'])
    for (y, x), label in [x[0] for x in sample['entities_tokenized']]:
        label = torch.argmax(label)

        lb_tensor_neg[0,y:x+1] = 0
        lb_tensor_pos[label][0,min(y+1, x):x+1] = 1

    hs_positives = [torch.masked_select(h_s_at_layer, x.bool()).view(-1, outputs.hidden_states[2].size(-1)) for x in lb_tensor_pos]
    hs_negatives = torch.masked_select(h_s_at_layer, lb_tensor_neg.bool()).view(-1, outputs.hidden_states[2].size(-1))
    
    return attention_positives, attention_negatives, hs_positives, hs_negatives

def label_sample(sample, llm, support_features, layer_id=9):
    
    input_ids = torch.tensor(sample['tokens'])

    outputs = llm(input_ids.to(llm.device).unsqueeze(0), output_attentions=True)
    
    h_s_at_layer = outputs.hidden_states[layer_id].detach().cpu().squeeze(0)

    attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()
    attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)

    hs_similarities = torch.cat([torch.max(torch.tensor(cosine_similarity(h_s_at_layer, x)), dim=-1).values.unsqueeze(0) for x in support['hs_pos']])
    hs_similarity_neg = torch.max(torch.tensor(cosine_similarity(h_s_at_layer, support['hs_neg'])), dim=-1).values.unsqueeze(0)

    
    tokenwise_pred = torch.argmax(torch.cat((hs_similarity_neg, hs_similarities), dim=0), dim=0)

    att_similarity_pos = torch.max(torch.tensor(cosine_similarity(attentions[0].view(-1, attentions.size(-1)), support['att_pos'])), dim=-1).values.view(attentions.size(-3), attentions.size(-2), 1)
    att_similarity_neg = torch.max(torch.tensor(cosine_similarity(attentions[0].view(-1, attentions.size(-1)), support['att_neg'])), dim=-1).values.view(attentions.size(-3), attentions.size(-2), 1)

    att_pred = torch.gt(att_similarity_pos, att_similarity_neg).float()

    #print([x[0][0] for x in sample['entities_tokenized'] if torch.argmax(x[0][1]) == positive_index])

    span_active = False
    span_start = -1
    predicted_spans = []
    for i in reversed(range(1, tokenwise_pred.size(0)-1)):
        if span_active and i >= span_start:
            continue

        #if not tokenizer.decode(sample['tokens'][i]).endswith(" ") and not tokenizer.decode(sample['tokens'][i+1])[0] in " '.,;:?!":
        #    continue
            
        elif span_active:
            span_active = False
        if tokenwise_pred[i] != 0 and torch.sum(att_similarity_pos[i+1][1:+2]) != 0:
            start, end = int(torch.argmax(att_similarity_pos[i+1][1:i+1]))+1, i
            # check if span is valid (or starts/ends within a word)
            span = tokenizer.decode(sample['tokens'][start:end+1])
            span_prev = tokenizer.decode(sample['tokens'][start-1:end+1])
            span_next = tokenizer.decode(sample['tokens'][start:end+2])

            if span[0] not in " '.,;:?!><()":
                if span_prev[:-len(span)][-1] not in " '.,;:?!><":
                    #print(f"discarding: '{span}'", span_prev)
                    continue

            if span[-1] not in " '.,;:?!><()":
                if span_next[len(span):][0] not in " '.,;:?!><":
                    #print(f"discarding: '{span}'", span_next)
                    continue
            
            span_active = True
            span_start = start
            predicted_spans.append((start, end, tokenwise_pred[i]))
            continue
    #print(att_pred.shape)
    #print(att_pred)

    return [x for x in reversed(predicted_spans)]

data = [x for x in dataset_gen]

true_seq = []
pred_seq = []

for episode in tqdm(range(n_episodes)):
    support_set, query_set = sample_episode(k, n_query, dataset_gen, types_index)
    # get support features
    features = {
        'att_pos': [],
        'att_neg': [],
        'hs_pos': [[] for _ in range(len(type_map))],
        'hs_neg': [],
    }
    for datasample in [data[x] for x in support_set]:
        att_pos, att_neg, hs_pos, hs_neg = get_features_for_sample(datasample, model, layer_id=layer_id, n_classes=len(type_map))
        features['att_pos'].append(att_pos)
        features['att_neg'].append(att_neg)
        for i in range(len(type_map)):
            features['hs_pos'][i].append(hs_pos[i])
        features['hs_neg'].append(hs_neg)

    support = {
        'att_pos': torch.cat(features['att_pos'], dim=0),
        'att_neg': torch.cat(features['att_neg'], dim=0),
        'hs_pos': [torch.cat(x, dim=0) for x in features['hs_pos'][1:]],
        'hs_neg': torch.cat(features['hs_neg'], dim=0),
    }
    #print(support['att_pos'].shape, support['att_neg'].shape, support['hs_pos'].shape, support['hs_neg'].shape)
    
    # classify queries
    for datasample in [data[x] for x in query_set]:
        predicted_spans = label_sample(datasample, model, support, layer_id=layer_id)
        #print([x[0][0] for x in datasample['entities_tokenized'] if torch.argmax(x[0][1]) == entity_type], predicted_spans)
        true_spans = [x[0][0] for x in datasample['entities_tokenized']]
        pred_spans = [span for span in predicted_spans]

        iob_true = ['O' for _ in datasample['tokens']]
        for span in datasample['entities_tokenized']:
            pred_type = torch.argmax(span[0][1])
            start, end = span[0][0]
            iob_true[start:end+1] = [f"I-{type_map[pred_type]}"] * (end+1-start)
            iob_true[start] = f"B-{type_map[pred_type]}"
            
        iob_pred = ['O' for _ in datasample['tokens']]
        for span in predicted_spans:
            start, end, pred_type = span
            iob_pred[start:end+1] = [f"I-{type_map[pred_type]}"] * (end+1-start)
            iob_pred[start] = f"B-{type_map[pred_type]}"

        true_seq.append(iob_true)
        pred_seq.append(iob_pred)

print(classification_report(true_seq, pred_seq, digits=4))


results = {
    'model_name': model_name,
    'dataset_id': dataset_id,
    'layer_id': layer_id,
    'k': k,
    'n_query': n_query,
    'n_episodes': n_episodes,
    'identifier': random_string,
}

precision = precision_score(true_seq, pred_seq)
precision_classwise = precision_score(true_seq, pred_seq, average=None)
recall = recall_score(true_seq, pred_seq)
recall_classwise = recall_score(true_seq, pred_seq, average=None)
f1 = f1_score(true_seq, pred_seq)
f1_classwise = f1_score(true_seq, pred_seq, average=None)

results["precision"] = precision
results["recall"] = recall
results["f1"] = f1

if len(f1_classwise) == len(type_map[1:]):
    for f1_value, p_value, r_value, classname in zip(f1_classwise, precision_classwise, recall_classwise, sorted(type_map[1:])):
        results[f"f1 ({classname})"] = f1_value
        results[f"precision ({classname})"] = p_value
        results[f"recall ({classname})"] = r_value

json.dump(results, open(f"result_fewshot_{dataset_id.split('/')[-1]}_{model_name.split('/')[-1]}_{layer_id}_{k}_{n_query}_{n_episodes}_{random_string}.json", "w"))
