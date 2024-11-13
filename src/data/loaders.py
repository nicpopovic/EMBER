from datasets import load_dataset
import random
from tqdm import tqdm
from src.data.util import Detokenizer
from torch.utils.data import Dataset
import os
import torch
import pickle
import json


class DataSample(object):

    def __init__(self, text, source_tokens, token_offsets, entities, relations, negative_mentions, ner_tags=None, source_token_annotations=None):
        self.text = text
        self.source_tokens = source_tokens
        self.token_offsets = token_offsets
        self.entities = entities
        self.relations = relations
        self.negative_mentions = negative_mentions
        self.ner_tags = ner_tags
        self.source_token_annotations = source_token_annotations

    def __repr__(self):
        return str({
            "text": self.text,
            "source_tokens": self.source_tokens,
            "ner_tags": self.ner_tags,
            "token_offsets": self.token_offsets,
            "entities": self.entities,
            "relations": self.relations,
            "negative_mentions": self.negative_mentions,
            "source_token_annotations": self.source_token_annotations,
        })

def loadDataset(dataset_id, split, dev=False, max_len_mention_negatives=5, skip_internal_spans=False):
    # loads dataset
    # valid ids: "conll2003", "retacred", "docred", "docred_jere"
    assert dataset_id in ["conll2003", "retacred", "docred", "docred_jere", "tner/ontonotes5", "tner/conll2003", "tner/wnut2017", "tner/multinerd", "tner/bc5cdr"]
    # valid splits: "train", "validation", "test"
    assert split in ["train", "validation", "test"]

    if dataset_id == "conll2003":
        return loadCoNLLdata("conll2003", split, dev=dev, max_len_mention_negatives=max_len_mention_negatives, skip_internal_spans=skip_internal_spans)
    elif dataset_id.startswith("tner/"):
        return loadTNERdata(dataset_id, split, dev=dev, max_len_mention_negatives=max_len_mention_negatives, skip_internal_spans=skip_internal_spans)
    elif dataset_id == "docred":
        if split == "train":
            split = "train_annotated"
        return loadDocREDdata("docred", split, dev=dev, max_len_mention_negatives=max_len_mention_negatives, skip_internal_spans=skip_internal_spans)
    elif dataset_id == "retacred":
        return loadReTACREDdata(split, dev=dev)

def ner_tags_to_annotations(ner_tags, tag_map, label_map):
    # function that parses ner_tags from dataset to spans
    # tag_map: translation of tags

    # we will collect spans here
    spans = []

    # keep track of active spans
    active_spans = {}
    for lb in label_map:
        active_spans[lb] = None

    # keep track of all spans and sub_spans
    positives = []

    # iterate over ner_tags and create spans
    for idx, tag in enumerate(ner_tags):

        #  A word with tag O is not part of a phrase, so we close all active spans
        if tag_map[tag] == 'O':
            for ent_type in active_spans.keys():
                if active_spans[ent_type] is not None:
                    # move span to output
                    spans.append((active_spans[ent_type], ent_type))
                    active_spans[ent_type] = None
            continue

        # get IB-tag and type
        if len(tag_map[tag].split("-")) > 2:
            b_i, ent_type = tag_map[tag].split("-")[0], "-".join(tag_map[tag].split("-")[1:])
        else:
            b_i, ent_type = tag_map[tag].split("-")
          

        if b_i == "I" and active_spans[ent_type] is None:
            print("Found I-tag without previous B-tag, overwriting to B-tag")
            b_i = "B"

        # track spans
        if b_i == "B":
            # marks start of a new span
            span = [idx, idx+1]
            # check for active span
            if active_spans[ent_type] is not None:
                # move previous span to output
                spans.append((active_spans[ent_type], ent_type))
            # save new span to active spans
            active_spans[ent_type] = span
            # save to positives
            positives.append([idx, idx+1])

        if b_i == "I" or f"I-{ent_type}" not in tag_map:
            # means that the word is inside a phrase

            # make sure there is an active span to close (should always be the case)
            #print(ent_type, tag_map[tag], ner_tags, active_spans)
            assert active_spans[ent_type] is not None

            # update active span
            active_spans[ent_type][1] = idx+1
            # save to positives
            positives.append([active_spans[ent_type][0], idx+1])
            for i in range(active_spans[ent_type][0]+1, idx+1):
                # add all subspans
                if [i, idx+1] not in positives:
                    positives.append([i, idx+1])

    # finally, save all active spans
    for ent_type in active_spans.keys():
        if active_spans[ent_type] is not None:
            # move previous span to output
            spans.append((active_spans[ent_type], ent_type))

    return spans

def vertexSetToAnnotations(vertexSet, sent_lengths):
    # converts vertexSet to spans of format ((start_token, end_token), class_label)
    positives = []
    cluster_map = []

    for c_i, ent_cluster in enumerate(vertexSet):
        for mention in ent_cluster:
            offset = 0
            
            sent_id = mention['sent_id']

            if sent_id != 0:
                offset += sum(sent_lengths[:sent_id])

            start_token_sent = mention['pos'][0] + offset
            end_token_sent = mention['pos'][1] + offset

            label = mention['type']
            cluster_map.append(c_i)
            positives.append([(start_token_sent, end_token_sent), label])

    return positives, cluster_map

def map_ner_ann_to_char(annotations, text, sent_len_tk, offsets):
    # change token index to character index
    new_annotations = []
    for ann in annotations:

        if ann[0][1] == sent_len_tk:
            end_index = len(text)
        else:
            end_index = offsets[ann[0][1]]
        new_span = [offsets[ann[0][0]], end_index]
        if ann[0][0] != 0:
            if text[new_span[0]] == " ":
                new_span[0] += 1
        new_annotations.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))

    return new_annotations

def loadCoNLLdata(huggingface_id, split, dev=False, max_len_mention_negatives=10, skip_internal_spans=False):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations
    dataset = load_dataset(huggingface_id)[split]
    
    data = []
    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset):
        if dev and len(data) >= 20:
            break
        
                        
        if len(sentence['ner_tags']) > 100:
            print(len(sentence['ner_tags']))
            continue
        
        text, offsets = detokenizer(sentence['tokens'], return_offsets=True)
        annotations = ner_tags_to_annotations(sentence['ner_tags'], tag_map=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], label_map=['PER', 'ORG', 'LOC', 'MISC'])

        negatives = get_negatives(annotations=annotations, sent_len_tk=len(sentence['tokens']), max_len=max_len_mention_negatives, skip_internal_spans=skip_internal_spans)

        token_annotations = annotations
        annotations = map_ner_ann_to_char(annotations=annotations, text=text, sent_len_tk=len(sentence['tokens']), offsets=offsets)

        annotations = [[ann] for ann in annotations]    # DataSample expects span annotations to be entity clusters - in this dataset each mention is an entity cluster

        batch = DataSample(text=text, source_tokens=[sentence['tokens']], token_offsets=offsets, entities=annotations, relations=[], negative_mentions=negatives, ner_tags=sentence['ner_tags'], source_token_annotations=token_annotations)
        data.append(batch)

    return data

def get_ner_maps(huggingface_id):

    if huggingface_id == "tner/ontonotes5":
        return ['O', 'B-CARDINAL', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON', 'B-NORP', 'B-GPE', 'I-GPE', 'B-LAW', 'I-LAW', 'B-ORG', 'I-ORG', 'B-PERCENT', 'I-PERCENT', 'B-ORDINAL', 'B-MONEY', 'I-MONEY', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-FAC', 'B-TIME', 'I-CARDINAL', 'B-LOC', 'B-QUANTITY', 'I-QUANTITY', 'I-NORP', 'I-LOC', 'B-PRODUCT', 'I-TIME', 'B-EVENT', 'I-EVENT', 'I-FAC', 'B-LANGUAGE', 'I-PRODUCT', 'I-ORDINAL', 'I-LANGUAGE'], ['CARDINAL', 'DATE', 'PERSON', 'NORP', 'GPE', 'LAW', 'ORG', 'PERCENT', 'ORDINAL', 'MONEY', 'WORK_OF_ART', 'FAC', 'TIME', 'LOC', 'QUANTITY', 'PRODUCT', 'EVENT', 'LANGUAGE']
    elif huggingface_id == "tner/conll2003":
        return ['O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC'], ['PER', 'ORG', 'LOC', 'MISC']
    elif huggingface_id == "tner/wnut2017":
        return ["O", "B-corporation","B-creative-work","B-group","B-location","B-person","B-product","I-corporation","I-creative-work","I-group","I-location","I-person","I-product"], ['corporation', 'creative-work', 'group', 'location', 'person', 'product']
    elif huggingface_id == "tner/bc5cdr":
        return ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], ['Chemical', 'Disease']
    elif huggingface_id == "tner/multinerd":
        return ['O', "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL", "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD", "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-PLANT", "I-PLANT", "B-MYTH", "I-MYTH", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI", "B-SUPER", "I-SUPER", "B-PHY", "I-PHY"], ['PER','LOC','ORG','ANIM','BIO','CEL','DIS','EVE','FOOD','INST','MEDIA','PLANT','MYTH','TIME','VEHI','SUPER','PHY']
    elif huggingface_id == "tner/conll2003":
        return ['O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC'], ['PER', 'ORG', 'LOC', 'MISC']
    elif huggingface_id == "conll2003":
        return ['O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC'], ['PER', 'ORG', 'LOC', 'MISC']
    else:
        raise NotImplementedError

def iob2_to_tob(iob2, tag_map, label_map):

    iob = [0 for _ in iob2]

    active_spans = {}
    for class_label in label_map:
        active_spans[class_label] = 0
    
    for i, tag in enumerate(iob2):
        if tag_map[tag] == 'O':
            # reset active spans
            for k in active_spans.keys():
                active_spans[k] = 0
            continue
        
        if len(tag_map[tag].split("-")) > 2:
            i_or_b, current_class_label = tag_map[tag].split("-")[0], "-".join(tag_map[tag].split("-")[1:])
        else:
            i_or_b, current_class_label = tag_map[tag].split("-")
            
        if i_or_b == "B":
            active_spans[current_class_label] = 1
            iob[i] = tag
            continue

        if i_or_b == "I":
            if active_spans[current_class_label] == 1:
                # all good
                iob[i] = tag
                continue
            else:
                active_spans[current_class_label] = 1
                # replace I tag with B tag
                iob[i] = tag_map.index(f"B-{current_class_label}")

    return iob

def loadTNERdata(huggingface_id, split, dev=False, max_len_mention_negatives=10, skip_internal_spans=False):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations
    if huggingface_id == "tner/multinerd":
        dataset = load_dataset(huggingface_id, 'en')[split]
    else:
        dataset = load_dataset(huggingface_id)[split]
    
    data = []
    max_len = 0
    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset):
        if dev and len(data) >= 20:
            break
                
        if len(sentence['tags']) > max_len:
            print(len(sentence['tags']))
            max_len = len(sentence['tags'])
        
        if huggingface_id == "tner/wnut2017":
            new_tags = []
            for x in sentence['tags']:
                if x == 12:
                    new_tags.append(0)
                else:
                    new_tags.append(x+1)
            sentence['tags'] = new_tags

            
        text, offsets = detokenizer(sentence['tokens'], return_offsets=True)


        tag_map, label_map = get_ner_maps(huggingface_id)
        #print(sentence['tokens'])
        iob2 = sentence['tags']

        iob = iob2_to_tob(sentence['tags'], tag_map, label_map)

        annotations = ner_tags_to_annotations(iob, tag_map=tag_map, label_map=label_map)
        negatives = get_negatives(annotations=annotations, sent_len_tk=len(sentence['tokens']), max_len=max_len_mention_negatives, skip_internal_spans=skip_internal_spans)

        annotations = map_ner_ann_to_char(annotations=annotations, text=text, sent_len_tk=len(sentence['tokens']), offsets=offsets)
        token_annotations = annotations

        annotations = [[ann] for ann in annotations]    # DataSample expects span annotations to be entity clusters - in this dataset each mention is an entity cluster

        batch = DataSample(text=text, source_tokens=[sentence['tokens']], token_offsets=offsets, entities=annotations, relations=[], negative_mentions=negatives, ner_tags=iob, source_token_annotations=token_annotations)
        data.append(batch)

    return data


def rebuild_cluster(anns, cluster_map):
    n_clusters = max(cluster_map)+1
    clusters = [[] for _ in range(n_clusters)]

    for ann, c_id in zip(anns, cluster_map):
        clusters[c_id].append(ann)
    
    return clusters

def get_negatives(annotations, sent_len_tk, max_len, skip_internal_spans=False):

    positives = [m[0] for m in annotations]

    blockers = [0 for _ in range(sent_len_tk)]
    for p in positives:
        for i in range(p[0],p[1]):
            blockers[i] = 1

    negatives = []
    for i in range(0, sent_len_tk):
        for j in range(i, min(sent_len_tk, i+max_len )):
            if sum(blockers[i:j+1]) == j - i + 1 and skip_internal_spans:
                continue
            if [i, j] not in positives and (i, j) not in positives:
                negatives.append([(i, j), "NONE"])

    return negatives

def loadDocREDdata(huggingface_id, split, dev=False, max_len_mention_negatives=10, skip_internal_spans=False):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations
    dataset = load_dataset(huggingface_id)[split]
    
    data = []
    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset, desc="parsing dataset"):
        if dev and len(data) >= 50:
            break

        flattened_tokens = [item for sublist in sentence['sents'] for item in sublist]

        text, offsets = detokenizer(flattened_tokens, return_offsets=True)

        # entities
        annotations, cluster_map = vertexSetToAnnotations(sentence['vertexSet'], [len(sublist) for sublist in sentence['sents']])
        negatives = get_negatives(annotations=annotations, sent_len_tk=len(flattened_tokens), max_len=max_len_mention_negatives, skip_internal_spans=skip_internal_spans)
        annotations = map_ner_ann_to_char(annotations=annotations, text=text, sent_len_tk=len(flattened_tokens), offsets=offsets)
        annotations = rebuild_cluster(annotations, cluster_map)

        # relations
        relations = []
        for h, t, r in zip(sentence['labels']['head'], sentence['labels']['tail'], sentence['labels']['relation_id']):
            relations.append(([h, t], r))

        batch = DataSample(text=text, source_tokens=sentence['sents'], token_offsets=offsets, entities=annotations, relations=relations, negative_mentions=negatives)
        data.append(batch)

    return data


def loadReTACREDdata(split, dev=False):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations

    splits = {
        "train": "train",
        "validation": "dev",
        "test": "test"
    }
    dataset = json.load(open(f"data/retacred/Re-TACRED/{splits[split]}.json", "rb"))
    
    data = []

    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset, desc="parsing dataset"):
        if dev and len(data) >= 50:
            break

        flattened_tokens = sentence['token']

        text, offsets = detokenizer(flattened_tokens, return_offsets=True)

        # entities
        annotations = []
        annotations.append([(sentence['subj_start'], sentence['subj_end']+1), sentence['subj_type']])
        annotations.append([(sentence['obj_start'], sentence['obj_end']+1), sentence['obj_type']])

        annotations = map_ner_ann_to_char(annotations=annotations, text=text, sent_len_tk=len(flattened_tokens), offsets=offsets)
        annotations = [[x] for x in annotations]

        # relations
        relations = [([0, 1], sentence["relation"])]

        batch = DataSample(text=text, source_tokens=flattened_tokens, token_offsets=offsets, entities=annotations, relations=relations, negative_mentions=[])
        data.append(batch)

    return data


class FeatureGenDataset(Dataset):
    # this is the dataset used for feature generation

    def __init__(self, huggingface_id, split, tokenizer=None, tokenizer_batch_size=1024, force_bos=True, negative_sampling=False, labels_as_tensors=True, rel=True, coref=True, **kwargs):
        super().__init__()
        self.samples = loadDataset(huggingface_id, split, **kwargs)

        self.labels_as_tensors = labels_as_tensors
        if labels_as_tensors:
            self.labelmaps = self.load_labelmaps(huggingface_id)

        self.rel = rel
        self.coref = coref
        self.tokenizer = tokenizer
        self.force_bos = force_bos
        self.negative_sampling = negative_sampling
        if tokenizer is not None:
            self.tokenize_data(batch_size=tokenizer_batch_size)
    
    def load_labelmaps(self, dataset_id):
        if dataset_id == "docred":
            return {
                "ner": ["NONE", "PER", "ORG", "LOC", "MISC", "TIME", "NUM"],
                "rel": ['NONE', 'P6', 'P17', 'P19', 'P20', 'P22', 'P25', 'P26', 'P27', 'P30', 'P31', 'P35', 'P36', 'P37', 'P39', 'P40', 'P50', 'P54', 'P57', 'P58', 'P69', 'P86', 'P102', 'P108', 'P112', 'P118', 'P123', 'P127', 'P131', 'P136', 'P137', 'P140', 'P150', 'P155', 'P156', 'P159', 'P161', 'P162', 'P166', 'P170', 'P171', 'P172', 'P175', 'P176', 'P178', 'P179', 'P190', 'P194', 'P205', 'P206', 'P241', 'P264', 'P272', 'P276', 'P279', 'P355', 'P361', 'P364', 'P400', 'P403', 'P449', 'P463', 'P488', 'P495', 'P527', 'P551', 'P569', 'P570', 'P571', 'P576', 'P577', 'P580', 'P582', 'P585', 'P607', 'P674', 'P676', 'P706', 'P710', 'P737', 'P740', 'P749', 'P800', 'P807', 'P840', 'P937', 'P1001', 'P1056', 'P1198', 'P1336', 'P1344', 'P1365', 'P1366', 'P1376', 'P1412', 'P1441', 'P3373']
            }
        elif dataset_id == "retacred":
            return {
                "ner": ["LOCATION",
                    "ORGANIZATION",
                    "PERSON",
                    "DATE",
                    "MONEY",
                    "PERCENT",
                    "TIME",
                    "CAUSE_OF_DEATH",
                    "CITY",
                    "COUNTRY",
                    "CRIMINAL_CHARGE",
                    "EMAIL",
                    "HANDLE",
                    "IDEOLOGY",
                    "NATIONALITY",
                    "RELIGION",
                    "STATE_OR_PROVINCE",
                    "TITLE",
                    "URL",
                    "NUMBER",
                    "ORDINAL",
                    "MISC",
                    "DURATION",
                    "O"],
                "rel": [
                    "no_relation",
                    "org:alternate_names",
                    "org:city_of_branch",
                    "org:country_of_branch",
                    "org:dissolved",
                    "org:founded",
                    "org:founded_by",
                    "org:member_of",
                    "org:members",
                    "org:number_of_employees/members",
                    "org:political/religious_affiliation",
                    "org:shareholders",
                    "org:stateorprovince_of_branch",
                    "org:top_members/employees",
                    "org:website",
                    "per:age",
                    "per:cause_of_death",
                    "per:charges",
                    "per:children",
                    "per:cities_of_residence",
                    "per:city_of_birth",
                    "per:city_of_death",
                    "per:countries_of_residence",
                    "per:country_of_birth",
                    "per:country_of_death",
                    "per:date_of_birth",
                    "per:date_of_death",
                    "per:employee_of",
                    "per:identity",
                    "per:origin",
                    "per:other_family",
                    "per:parents",
                    "per:religion",
                    "per:schools_attended",
                    "per:siblings",
                    "per:spouse",
                    "per:stateorprovince_of_birth",
                    "per:stateorprovince_of_death",
                    "per:stateorprovinces_of_residence",
                    "per:title"
                ]
            }
        elif dataset_id == "conll2003":
            return {
                "ner": ["NONE", "PER", "ORG", "LOC", "MISC"],
                "iob": ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            }
        elif dataset_id.startswith("tner/"):
            return {
                "ner": ["NONE"] + get_ner_maps(dataset_id)[1],
            }

    def tokenize_data(self, batch_size):
        discarded_samples = 0

        period_token = self.tokenizer.encode(".")[0]
        period_token = self.tokenizer.eos_token_id

        fixed_front = 0
        fixed_back = 0
        for i in tqdm(range(0, len(self.samples), batch_size), desc="tokenizing"):
            #inputs = self.tokenizer.batch_encode_plus([x.text for x in self.samples[i:i+batch_size]], return_offsets_mapping=True)


            #for j, (ids, offsets) in enumerate(zip(inputs.input_ids, inputs.offset_mapping)):
            for j, texts in enumerate([x.text for x in self.samples[i:i+batch_size]]):
                inputs = self.tokenizer(texts, return_offsets_mapping=True)
                ids, offsets = inputs.input_ids, inputs.offset_mapping

                if ids[-1] != period_token:
                    ids = ids + [period_token]

                # override occasional tokenizer weirdness in pythia caused by foreign language symbols
                prev = 0
                new_offsets = []
                for offset in offsets:
                    if prev < offset[0]-1:
                        print(offsets)
                        print(prev, offset)
                        print(offset[0], prev+1, self.samples[i+j].text[offset[0]:offset[1]], "->", self.samples[i+j].text[prev+1:offset[1]])
                        new_offsets.append((prev+1, offset[1]))
                    else:
                        new_offsets.append(offset)
                    prev = offset[1]
                offsets = new_offsets

                idx = i + j

                # enforce bos token
                token_offset = 0
                if self.force_bos:
                    bos_token_id = self.tokenizer.bos_token_id
                    if ids[0] != bos_token_id:
                        token_offset = 1
                        ids = [bos_token_id] + ids

                # write input ids to samples
                self.samples[idx].tokens = ids

                # adjust annotation indexing
                self.samples[idx].entities_tokenized = []
                self.samples[idx].corefs = None
                self.samples[idx].token_offsets_model = offsets

                token_spans = []
                for entity in self.samples[idx].entities:
                    mentions = []
                    for annotation in entity:

                        # map start character to token
                        start_char = annotation[0][0]
                        start_token = -1
                        for k, tk in enumerate(offsets):
                            # skip bos token
                            if tk == (0, 0):
                                continue
                            if tk[0] <= start_char and tk[1] >= start_char:
                                start_token = k + token_offset
                                break

                        
                        try:
                            # make sure a start token was set
                            assert start_token != -1
                        except:
                            print(annotation, offsets, self.force_bos, self.tokenizer.bos_token_id)
                            raise

                        # map start character to token
                        end_char = annotation[0][1]
                        end_token = -1
                        for k, tk in enumerate(offsets):
                            # skip bos token
                            if tk == (0, 0):
                                continue
                            if tk[0] <= end_char and tk[1] >= end_char:
                                end_token = k + token_offset
                                break
                        
                        # make sure an end token was set
                        assert start_token != -1

                        # fix misaligned token spans occuring when entity is not preceeded by whitespace
                        surface_form = annotation[-1].strip()
                        surface_form_tokenized = self.tokenizer.decode(ids[start_token:end_token+1]).strip()
                        if surface_form != surface_form_tokenized:
                            surface_form_tokenized = self.tokenizer.decode(ids[start_token+1:end_token+1]).strip()
                            if surface_form == surface_form_tokenized:
                                fixed_front += 1
                                start_token += 1
                            surface_form_tokenized = self.tokenizer.decode(ids[start_token:end_token]).strip()
                            if surface_form == surface_form_tokenized:
                                fixed_back += 1
                                end_token -= 1

                        if str((start_token, end_token)) not in token_spans:
                            token_spans.append(str((start_token, end_token)))
                            # append
                            mentions.append(((start_token, end_token), annotation[1], ids[start_token:end_token+1]))
                        else:
                            discarded_samples += 1
                    self.samples[idx].entities_tokenized.append(mentions)

                self.samples[idx].negative_mentions_tokenized = []

                for annotation in self.samples[idx].negative_mentions:
                    # map start character to token
                    start_char = annotation[0][0]
                    start_token = -1
                    for k, tk in enumerate(offsets):
                        # skip bos token
                        if tk == (0, 0):
                            continue
                        if tk[0] <= start_char and tk[1] >= start_char:
                            start_token = k + token_offset
                            break
                    
                    # make sure a start token was set
                    assert start_token != -1

                    # map start character to token
                    end_char = annotation[0][1]
                    end_token = -1
                    for k, tk in enumerate(offsets):
                        # skip bos token
                        if tk == (0, 0):
                            continue
                        if tk[0] <= end_char and tk[1] >= end_char:
                            end_token = k + token_offset
                            break
                    
                    # make sure an end token was set
                    assert start_token != -1

                    if str((start_token, end_token)) not in token_spans:
                        token_spans.append(str((start_token, end_token)))
                        # append
                        self.samples[idx].negative_mentions_tokenized.append(((start_token, end_token), annotation[1], ids[start_token:end_token+1]))
                    else:
                        discarded_samples += 1

        print(fixed_front, fixed_back)

        if discarded_samples > 0:
            pass # deprecated
            print(f"{discarded_samples} samples were discarded due to identical spans after tokenization")

    def sample_negatives(self, idx):
        negative_mentions, negative_corefs, negative_relations = [], [], []

        # sample negative mentions
        if self.negative_sampling["mentions"]["type"] in ["random", "all"]:

            max_len_for_negatives = self.negative_sampling["mentions"]["max_len"]

            positives = [item[0] for sublist in self.samples[idx].entities_tokenized for item in sublist]

            # generate all negatives
            negatives = []
            for i in range(len(self.samples[idx].tokens)):
                if self.force_bos and i == 0:
                    # skip bos_token
                    continue
                for b in range(max(0, i-max_len_for_negatives), i):
                    if [b, i] in positives:
                        continue
                    if self.labels_as_tensors:
                        lb_t = torch.zeros((len(self.labelmaps['ner'])))
                        lb_t[0] = 1
                        negatives.append(([b, i], lb_t))
                    else:
                        negatives.append(([b, i], 'NONE'))
            
            if self.negative_sampling["mentions"]["type"] == "random":
                negative_mentions = random.sample(negatives, min(len(negatives), self.negative_sampling["mentions"]["n_samples"]))
            elif self.negative_sampling["mentions"]["type"] == "all":
                negative_mentions = negatives

        # sample negative corefs
        if self.negative_sampling["coref"]["type"] in ["random", "all"]:

            max_len_for_negatives = self.negative_sampling["mentions"]["max_len"]

            all_mentions = [item for sublist in self.samples[idx].entities_tokenized for item in sublist]

            corefs = []
            for coref_cluster in self.samples[idx].entities_tokenized:
                for ent_a in coref_cluster:
                    for ent_b in all_mentions:
                        if ent_a == ent_b or ent_b in coref_cluster:
                            continue
                        if ent_a[0][1] < ent_b[0][0]:
                            if self.labels_as_tensors:
                                coref = ([ent_a[0], ent_b[0]], torch.Tensor([1,0]))
                            else:
                                coref = ([ent_a[0], ent_b[0]], "NO_COREF")
                            if coref not in corefs:
                                corefs.append(coref)
                
            if self.negative_sampling["coref"]["type"] == "random":
                negative_corefs = random.sample(corefs, min(len(negatives), self.negative_sampling["coref"]["n_samples"]))
            elif self.negative_sampling["coref"]["type"] == "all":
                negative_corefs = corefs

        # sample negative relations
        if self.negative_sampling["relations"]["type"] in ["random", "all"]:

            positives = [item[0] for item in self.samples[idx].relations]

            # generate all negatives
            negatives = []
            for i in range(len(self.samples[idx].entities_tokenized)):
                for j in range(len(self.samples[idx].entities_tokenized)):
                    if [i, j] in positives:
                        continue
                    if self.labels_as_tensors:
                        lb_t = torch.zeros((len(self.labelmaps['rel'])))
                        lb_t[0] = 1
                        negatives.append(([j, i], lb_t))
                    else:
                        negatives.append(([j, i], 'NONE'))
            
            if self.negative_sampling["relations"]["type"] == "random":
                negative_relations = random.sample(negatives, min(len(negatives), self.negative_sampling["relations"]["n_samples"]))
            elif self.negative_sampling["relations"]["type"] == "all":
                negative_relations = negatives


        return negative_mentions, negative_corefs, negative_relations

    def parse_corefs(self, idx):


        corefs = []
        for coref_cluster in self.samples[idx].entities_tokenized:
            for ent_a in coref_cluster:
                for ent_b in coref_cluster:
                    if ent_a == ent_b:
                        continue
                    if ent_a[0][1] < ent_b[0][0]:
                        if self.labels_as_tensors:
                            coref = ([ent_a[0], ent_b[0]], torch.Tensor([0,1]))
                        else:
                            coref = ([ent_a[0], ent_b[0]], "COREF")
                        if coref not in corefs:
                            corefs.append(coref)

        self.samples[idx].corefs = corefs
    
    def convert_relations(self, idx):
        # convert relations to tensors
        to_tensors = []
        for rel in self.samples[idx].relations:
            lb_t = torch.zeros((len(self.labelmaps['rel'])))
            lb_t[self.labelmaps['rel'].index(rel[1])] = 1
            to_tensors.append((rel[0], lb_t))
        
        # merge labels if one pair has multiple
        pair_dict = {}
        for rel in to_tensors:
            if str(rel[0]) not in pair_dict.keys():
                pair_dict[str(rel[0])] = rel
            else:
                lb1 = pair_dict[str(rel[0])][1]
                lb_new = lb1 + rel[1]
                pair_dict[str(rel[0])] = (rel[0], lb_new)
        
        return [*pair_dict.values()]
    
    def convert_entities(self, idx):
        # convert relations to tensors
        to_tensors = []
        for ent in self.samples[idx].entities_tokenized:
            mentions = []
            for mention in ent:
                lb_t = torch.zeros((len(self.labelmaps['ner'])))
                lb_t[self.labelmaps['ner'].index(mention[1])] = 1
                mentions.append((mention[0], lb_t))

            to_tensors.append(mentions)

        return to_tensors   
     
    def convert_mentions(self, idx):
        # convert mentions to tensors
        to_tensors = []
        for mention in self.samples[idx].negative_mentions_tokenized:
            lb_t = torch.zeros((len(self.labelmaps['ner'])))
            lb_t[self.labelmaps['ner'].index(mention[1])] = 1
            to_tensors.append((mention[0], lb_t))

        
        return to_tensors

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        if self.tokenizer is None:
            sample = {
                'text':self.samples[idx].text,
                'entities':self.samples[idx].entities,
                'relations':self.samples[idx].relations,
            }
        else:
            
            if self.labels_as_tensors:
                if self.rel:
                    converted_relations = self.convert_relations(idx)
                else:
                    converted_relations = self.samples[idx].relations
                converted_entities = self.convert_entities(idx)
                converted_mentions = self.convert_mentions(idx)
            else:
                converted_relations = self.samples[idx].relations
                converted_entities = self.samples[idx].entities_tokenized
                converted_mentions = self.samples[idx].negative_mentions_tokenized
            _, negative_corefs, negative_relations = [], [], []
            if self.samples[idx].corefs is None:
                if self.coref:
                    self.parse_corefs(idx)
            if self.negative_sampling:
                if self.coref or self.rel:
                    _, negative_corefs, negative_relations = self.sample_negatives(idx)


            sample = {
                'text':self.samples[idx].text,
                'tokens':self.samples[idx].tokens,
                'source_tokens':self.samples[idx].source_tokens,
                'token_offsets':self.samples[idx].token_offsets,
                'token_offsets_model':self.samples[idx].token_offsets_model,
                'ner_tags':self.samples[idx].ner_tags,
                'entities':self.samples[idx].entities,
                'entities_tokenized':converted_entities,
                'corefs': self.samples[idx].corefs,
                'relations':converted_relations,
                'negative_mentions':converted_mentions,
                'negative_corefs':negative_corefs,
                'negative_relations':negative_relations,
                'source_token_annotations':self.samples[idx].source_token_annotations,
                
            }
        return sample



class DatasetFromCache(Dataset):

    def __init__(self, path_to_data, eval_helper=None):
        super(DatasetFromCache, self).__init__()
        self.load(path_to_data)
        self.eval_helper_dataset = eval_helper
    
    def load(self, path_to_data):

        # LOAD NER DATA
        ft_ner = torch.load(os.path.join(path_to_data, "ft_ner.pt"))
        lb_ner = torch.load(os.path.join(path_to_data, "lb_ner.pt"))

        with open(os.path.join(path_to_data, f"index_ner.pt"), 'rb') as f:
            index_ner = pickle.load(f)

        ft_ner_by_doc = [[] for _ in range(max(index_ner)+1)]
        lb_ner_by_doc = [[] for _ in range(max(index_ner)+1)]

        for ft, lb, idx in tqdm(zip(ft_ner, lb_ner, index_ner)):
            ft_ner_by_doc[idx].append(ft)
            lb_ner_by_doc[idx].append(lb)
        
        self.features_ner = ft_ner_by_doc
        self.labels_ner = lb_ner_by_doc

    def __len__(self):
        return len(self.features_ner)
        
    def __getitem__(self, idx):
        if self.eval_helper_dataset is not None:
            # mention annotation
            positive_mentions = [item for sublist in self.eval_helper_dataset[idx]['entities_tokenized'] for item in sublist]
            negative_mentions = self.eval_helper_dataset[idx]['negative_mentions']
            all_mentions = positive_mentions + negative_mentions
            return {
                'features_ner':torch.stack(self.features_ner[idx]),
                'labels_ner':torch.stack(self.labels_ner[idx]),
                'token_offsets_model':self.eval_helper_dataset[idx]['token_offsets_model'],
                'token_offsets':self.eval_helper_dataset[idx]['token_offsets'],
                'text':self.eval_helper_dataset[idx]['text'],
                'ner_tags':self.eval_helper_dataset[idx]['ner_tags'],
                'annotations_mentions':all_mentions,
                'entities':self.eval_helper_dataset[idx]['source_token_annotations'],
            }
       
        return {
            'features_ner':torch.stack(self.features_ner[idx]),
            'labels_ner':torch.stack(self.labels_ner[idx]),
        }


class DatasetFromCacheRE(Dataset):

    def __init__(self, path_to_data, feature_type):
        super(DatasetFromCacheRE, self).__init__()
        self.feature_type = feature_type
        self.load(path_to_data)
    
    def load(self, path_to_data):

        # LOAD NER DATA
        ft_rel = torch.load(os.path.join(path_to_data, f"ft_rel_{self.feature_type}.pt"))
        lb_rel = torch.load(os.path.join(path_to_data, "lb_rel.pt"))

        with open(os.path.join(path_to_data, f"index_rel.pt"), 'rb') as f:
            index_rel = pickle.load(f)

        ft_rel_by_doc = [[] for _ in range(max(index_rel)+1)]
        lb_rel_by_doc = [[] for _ in range(max(index_rel)+1)]

        for ft, lb, idx in tqdm(zip(ft_rel, lb_rel, index_rel)):
            ft_rel_by_doc[idx].append(ft)
            lb_rel_by_doc[idx].append(lb)
        
        self.features_rel = ft_rel_by_doc
        self.labels_rel = lb_rel_by_doc

    def __len__(self):
        return len(self.features_rel)
        
    def __getitem__(self, idx):

        return {
            'features_rel':torch.stack(self.features_rel[idx]),
            'labels_rel':torch.stack(self.labels_rel[idx]),
        }

class JSONDataset(Dataset):

    def __init__(self, path="conll2003_generated_validation_full_annotated.json"):
        super().__init__()
        
        self.samples = json.load(open(path, "r"))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]
