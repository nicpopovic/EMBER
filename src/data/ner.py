from datasets import load_dataset
import random
from tqdm import tqdm
from src.data.util import Detokenizer
from src.util import get_iob_labels
from src.data.loaders import get_ner_maps
from torch.utils.data import Dataset
import os
import torch
import torch.nn.functional as F


def load_labelmaps(dataset_id):
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
            "iob": get_ner_maps(dataset_id)[0],
        }


class NERFeatureGenDataset(Dataset):

    def __init__(self, huggingface_id, split, tokenizer=None, tokenizer_batch_size=1024, force_bos=True, **kwargs):
        super().__init__()
        if huggingface_id == "conll2003":
            self.samples = loadCoNLLdata(huggingface_id, split, **kwargs)
        elif huggingface_id == "docred":
            self.samples = loadDocREDdata(huggingface_id, split, **kwargs)
        elif huggingface_id == "docred_coref":
            self.samples = loadDocREDdataCoref("docred", split, **kwargs)
        else:
            raise NotImplementedError(f"Methods for '{huggingface_id}' not implemented!")
        self.tokenizer = tokenizer
        self.force_bos = force_bos
        if tokenizer is not None:
            self.tokenize_data(batch_size=tokenizer_batch_size)

    def tokenize_data(self, batch_size):
        discarded_samples = 0
        for i in tqdm(range(0, len(self.samples), batch_size), desc="tokenizing"):
            inputs = self.tokenizer.batch_encode_plus([x.text for x in self.samples[i:i+batch_size]], return_offsets_mapping=True)

            
            for j, (ids, offsets) in enumerate(zip(inputs.input_ids, inputs.offset_mapping)):
                idx = i + j

                # enforce bos token
                token_offset = 0
                if self.force_bos:
                    if ids[0] != self.tokenizer.bos_token_id:
                        token_offset = 1
                        ids = [self.tokenizer.bos_token_id] + ids

                # write input ids to samples
                self.samples[idx].tokens = ids

                # adjust annotation indexing
                self.samples[idx].annotations_tokenized = []
                token_spans = []
                for annotation in self.samples[idx].annotations:
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
                        self.samples[idx].annotations_tokenized.append(((start_token, end_token), annotation[1], ids[start_token:end_token+1]))
                    else:
                        discarded_samples += 1

    
        if discarded_samples > 0:
            print(f"{discarded_samples} samples were discarded due to identical spans after tokenization")


    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        if self.tokenizer is None:
            sample = {
                'text':self.samples[idx].text,
                'annotations':self.samples[idx].annotations,
            }
        else:
            sample = {
                'text':self.samples[idx].text,
                'tokens':self.samples[idx].tokens,
                'annotations':self.samples[idx].annotations,
                'annotations_tokenized':self.samples[idx].annotations_tokenized,
            }
        return sample


class DataSampleNERBatch(object):
    # data object containing batch of annotations for a sentence

    def __init__(self, text:str, positives:list, negatives:list, negative_sampling="balanced_random"):
        self.text = text
        self.annotations = self.select_annotations(positives, negatives, method=negative_sampling)
        self.tokens = None
        self.annotations_tokenized = None

    def select_annotations(self, positives, negatives, method="balanced_random"):
        # generate mix of positive and negative labels according to selected method
        if method == "balanced_random":
            if len(negatives) < len(positives):
                return positives + negatives
            return positives + random.sample(negatives, max(len(positives), 1))
        elif method == "no_sampling":
            return positives + negatives
        elif method == "random20":
            if len(negatives) < len(positives):
                return positives + negatives
            return positives + random.sample(negatives, min(len(negatives), 20))
        else:
            raise NotImplementedError(f"Method '{method}' not implemented!")


def ner_tags_to_annotations(ner_tags, max_len_for_negatives=99):
    # function that parses ner_tags from dataset to spans
    # translation of tags
    tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    # we will collect spans here
    spans = []

    # keep track of active spans
    active_spans = {
        "PER": None,
        "ORG": None,
        "LOC": None,
        "MISC": None,
    }

    # keep track of all spans and sub_spans
    positives = []

    # iterate over ner_tags and create spans
    for idx, tag in enumerate(ner_tags):

        #  A word with tag O is not part of a phrase, so we close all active spans
        if tag == 0:
            for ent_type in active_spans.keys():
                if active_spans[ent_type] is not None:
                    # move span to output
                    spans.append((active_spans[ent_type], ent_type))
                    active_spans[ent_type] = None
            continue

        # get IB-tag and type
        b_i, ent_type = tags[tag].split("-")

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

        if b_i == "I":
            # means that the word is inside a phrase

            # make sure there is an active span to close (should always be the case)
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
    
    # generate all negatives
    negatives = []
    for i in range(len(ner_tags)+1):
        for b in range(max(0, i-max_len_for_negatives), i):
            if [b, i] in positives:
                continue
            negatives.append(([b, i], 'NONE'))

    return spans, negatives


def loadCoNLLdata(huggingface_id, split, max_len_for_negatives=99, dev=False, negative_sampling="balanced_random"):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations
    dataset = load_dataset(huggingface_id)[split]
    
    data = []
    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset):
        if dev and len(data) >= 20:
            break
        text, offsets = detokenizer(sentence['tokens'], return_offsets=True)
        
        annotations, negatives = ner_tags_to_annotations(sentence['ner_tags'], max_len_for_negatives=max_len_for_negatives)

        # change token index to character index
        new_annotations, new_negatives = [], []
        for ann in annotations:

            if ann[0][1] == len(sentence['tokens']):
                end_index = len(text)
            else:
                end_index = offsets[ann[0][1]]
            new_span = [offsets[ann[0][0]], end_index]
            if ann[0][0] != 0:
                if text[new_span[0]] == " ":
                    new_span[0] += 1
            new_annotations.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))
            """
            try:
                pre = detokenizer(sentence['tokens'][ann[0][0]:ann[0][1]])
                post = text[new_span[0]:new_span[1]]
                assert pre == post
            except AssertionError:
                print(text)
                print(f"'{post}' != '{pre}'")
            """
        for ann in negatives:

            if ann[0][1] == len(sentence['tokens']):
                end_index = len(text)
            else:
                end_index = offsets[ann[0][1]]
            new_span = [offsets[ann[0][0]], end_index]
            if ann[0][0] != 0:
                if text[new_span[0]] == " ":
                    new_span[0] += 1
            new_negatives.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))
        annotations = new_annotations
        negatives = new_negatives

        batch = DataSampleNERBatch(text=text, positives=annotations, negatives=negatives, negative_sampling=negative_sampling)
        data.append(batch)

    return data


def vertexSetToAnnotations(vertexSet, sent_lengths, max_len_for_negatives=99):
    # converts vertexSet to spans of format ((start_token, end_token), class_label)
    positives = []

    for ent_cluster in vertexSet:
        for mention in ent_cluster:
            offset = 0
            
            sent_id = mention['sent_id']

            if sent_id != 0:
                offset += sum(sent_lengths[:sent_id])

            start_token_sent = mention['pos'][0] + offset
            end_token_sent = mention['pos'][1] + offset

            label = mention['type']

            positives.append([(start_token_sent, end_token_sent), label])

    # generate all negatives
    negatives = []
    for i in range(sum(sent_lengths)):
        for b in range(max(0, i-max_len_for_negatives), i):
            if [b, i] in positives:
                continue
            negatives.append(([b, i], 'NONE'))

    return positives, negatives


def loadDocREDdata(huggingface_id, split, max_len_for_negatives=99, dev=False, negative_sampling="balanced_random"):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations
    dataset = load_dataset(huggingface_id)[split]
    
    data = []
    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset):
        if dev and len(data) >= 20:
            break

        flattened_tokens = [item for sublist in sentence['sents'] for item in sublist]

        text, offsets = detokenizer(flattened_tokens, return_offsets=True)

        annotations, negatives = vertexSetToAnnotations(sentence['vertexSet'], [len(sublist) for sublist in sentence['sents']], max_len_for_negatives=max_len_for_negatives)

        #for annotation in annotations:
            #print(flattened_tokens[annotation[0][0]:annotation[0][1]])
        # change token index to character index
        new_annotations, new_negatives = [], []
        for ann in annotations:

            if ann[0][1] == len(flattened_tokens):
                end_index = len(text)
            else:
                end_index = offsets[ann[0][1]]
            new_span = [offsets[ann[0][0]], end_index]
            if ann[0][0] != 0:
                if text[new_span[0]] == " ":
                    new_span[0] += 1
            new_annotations.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))
            """
            try:
                pre = detokenizer(sentence['tokens'][ann[0][0]:ann[0][1]])
                post = text[new_span[0]:new_span[1]]
                assert pre == post
            except AssertionError:
                print(text)
                print(f"'{post}' != '{pre}'")
            """
        for ann in negatives:

            if ann[0][1] == len(flattened_tokens):
                end_index = len(text)
            else:
                end_index = offsets[ann[0][1]]
            new_span = [offsets[ann[0][0]], end_index]
            if ann[0][0] != 0:
                if text[new_span[0]] == " ":
                    new_span[0] += 1
            new_negatives.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))
        annotations = new_annotations
        negatives = new_negatives

        batch = DataSampleNERBatch(text=text, positives=annotations, negatives=negatives, negative_sampling=negative_sampling)
        data.append(batch)

    return data


class BasicTrainingDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.features, self.labels = None, None
        self.load()

    def load(self):
        print(f"loading from {os.path.join(self.path, f'features.pt')}")
        self.features = torch.load(os.path.join(self.path, f"features.pt"), map_location=torch.device("cpu"))
        if torch.isnan(self.features).any().item():
            print("Warning: found nan values in features! Replacing with zeros...")
            self.features = torch.nan_to_num(self.features)
        self.labels = torch.load(os.path.join(self.path, f"labels.pt"), map_location=torch.device("cpu"))
    
    def __len__(self):
        return self.features.size(0)
        
    def __getitem__(self, idx):

        
        if self.labels[idx] == 0:
            label_value = torch.zeros_like(self.labels[idx])
        else:
            label_value = torch.ones_like(self.labels[idx])
        
        label_value = self.labels[idx]

        return {
            'data':self.features[idx],
            'label':label_value
        }


def vertexSetToCorefClusters(vertexSet, sent_lengths, max_len_for_negatives=99):
    # converts vertexSet to spans of format ((start_token, end_token), class_label)
    positives = []

    all_mentions = []
    for ent_cluster in vertexSet:

        mentions = []
        for mention in ent_cluster:
            offset = 0
            
            sent_id = mention['sent_id']

            if sent_id != 0:
                offset += sum(sent_lengths[:sent_id])

            start_token_sent = mention['pos'][0] + offset
            end_token_sent = mention['pos'][1] + offset

            mentions.append((start_token_sent, end_token_sent))

        all_mentions.extend(mentions)
        for ment_a in mentions:
            for ment_b in mentions:
                if ment_a == ment_b:
                    continue
                if ment_a[0] > ment_b[1]:
                    positives.append([(ment_b[1], ment_a[0]+1), 'COREF'])
                    
    # generate all negatives
    negatives = []

    for ment_a in all_mentions:
        for ment_b in all_mentions:
            if ment_a == ment_b:
                continue
            if ment_a[0] > ment_b[1] and [(ment_b[1], ment_a[0]+1), 'COREF'] not in positives:
                negatives.append([(ment_b[1], ment_a[0]+1), 'NO_COREF'])

    return positives, negatives


def loadDocREDdataCoref(huggingface_id, split, max_len_for_negatives=99, dev=False, negative_sampling="balanced_random"):
    # returns parsed data
    detokenizer = Detokenizer()
    # loads huggingface dataset and converts it to individual annotations
    dataset = load_dataset(huggingface_id)[split]
    
    data = []
    # iterate over sentences and convert them to individual annotations
    for sentence in tqdm(dataset):
        if dev and len(data) >= 20:
            break

        flattened_tokens = [item for sublist in sentence['sents'] for item in sublist]

        text, offsets = detokenizer(flattened_tokens, return_offsets=True)

        annotations, negatives = vertexSetToCorefClusters(sentence['vertexSet'], [len(sublist) for sublist in sentence['sents']], max_len_for_negatives=max_len_for_negatives)

        #for annotation in annotations:
            #print(flattened_tokens[annotation[0][0]:annotation[0][1]])
        # change token index to character index
        new_annotations, new_negatives = [], []
        for ann in annotations:

            if ann[0][1] == len(flattened_tokens):
                end_index = len(text)
            else:
                end_index = offsets[ann[0][1]]
            new_span = [offsets[ann[0][0]], end_index]
            if ann[0][0] != 0:
                if text[new_span[0]] == " ":
                    new_span[0] += 1
            new_annotations.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))
            """
            try:
                pre = detokenizer(sentence['tokens'][ann[0][0]:ann[0][1]])
                post = text[new_span[0]:new_span[1]]
                assert pre == post
            except AssertionError:
                print(text)
                print(f"'{post}' != '{pre}'")
            """
        for ann in negatives:

            if ann[0][1] == len(flattened_tokens):
                end_index = len(text)
            else:
                end_index = offsets[ann[0][1]]
            new_span = [offsets[ann[0][0]], end_index]
            if ann[0][0] != 0:
                if text[new_span[0]] == " ":
                    new_span[0] += 1
            new_negatives.append((new_span, *ann[1:], text[new_span[0]:new_span[1]]))
        annotations = new_annotations
        negatives = new_negatives

        batch = DataSampleNERBatch(text=text, positives=annotations, negatives=negatives, negative_sampling=negative_sampling)
        data.append(batch)

    return data


def get_IOB_per_token(sample, dataset_id):
    labelmaps = load_labelmaps(dataset_id)

    ner_tags = [0] * len(sample['tokens'])

    for ent in sample['entities_tokenized']:
        for mention in ent:
            entity_type = labelmaps["ner"][torch.argmax(mention[1]).item()]
            b_token = labelmaps["iob"].index("B-"+entity_type)
            i_token = labelmaps["iob"].index("I-"+entity_type)
            
            ner_tags[mention[0][0]:mention[0][1]+1] = [i_token] * (mention[0][1]+1-mention[0][0])
            ner_tags[mention[0][0]] = b_token
    
    return ner_tags


class CollateManagerExp1(object):

    def __init__(self, dataset_id) -> None:
        self.dataset_id = dataset_id

    def collateExp1(self, batch):
        sequences = [torch.tensor(x['tokens']) for x in batch]
        masks = [torch.tensor([False]+([True]*(len(x)-1))) for x in sequences]
        ner_tags = []

        for x in batch:
            nt = get_IOB_per_token(x, self.dataset_id)
            ner_tags.append(torch.tensor(nt))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) # pad token does not matter (autoregressive)
        ner_tags_batch = torch.nn.utils.rnn.pad_sequence(ner_tags, batch_first=True, padding_value=0)
        label_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)

        # mention annotation
        positive_mentions = [[item for sublist in x['entities_tokenized'] for item in sublist] for x in batch]
        negative_mentions = [x['negative_mentions'] for x in batch]
        all_mentions = [a+b for a,b in zip(positive_mentions, negative_mentions)]

        return {
            'input_ids': input_ids,
            'labels': ner_tags_batch,
            'label_mask': label_mask.unsqueeze(-1)
        }
    
    def collateExp2(self, batch):
        sequences = [torch.tensor(x['tokens']) for x in batch]
        masks = [torch.tensor([False]+([True]*(len(x)-1))) for x in sequences]
        ner_tags = []

        for x in batch:
            nt = get_IOB_per_token(x, self.dataset_id)
            ner_tags.append(torch.tensor(nt))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) # pad token does not matter (autoregressive)
        ner_tags_batch = torch.nn.utils.rnn.pad_sequence(ner_tags, batch_first=True, padding_value=0)
        label_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)

        # mention annotation
        positive_mentions = [[item for sublist in x['entities_tokenized'] for item in sublist] for x in batch]
        negative_mentions = [x['negative_mentions'] for x in batch]
        entities = [x['entities'] for x in batch]
        all_mentions = [a+b for a,b in zip(positive_mentions, negative_mentions)]

        return {
            'input_ids': input_ids,
            'labels': ner_tags_batch,
            'entities': positive_mentions,
            'label_mask': label_mask.unsqueeze(-1)
        }

    def collateExp4(self, batch):
        sequences = [torch.tensor(x['tokens']) for x in batch]
        masks = [torch.tensor([False]+([False]*(len(x)-1))) for x in sequences]
        ner_tags = []

        for i, x in enumerate(batch):
            nt = torch.zeros_like(torch.Tensor(x['tokens']))
            for entity, _ in zip([item for sublist in x['entities_tokenized'] for item in sublist], x['entities']):
                if entity[0][0] == entity[0][1]:
                    masks[i][entity[0][0]:entity[0][1]+2] = True
                    continue
                nt[entity[0][0]+1:entity[0][1]+1] = 1
                masks[i][entity[0][0]:entity[0][1]+2] = True
                #print(tokenizer.decode(x['tokens'][entity[0][0]+1:entity[0][1]+1]), _)

            ner_tags.append(nt.clone().detach())
        
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) # pad token does not matter (autoregressive)
        ner_tags_batch = torch.nn.utils.rnn.pad_sequence(ner_tags, batch_first=True, padding_value=0)
        label_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)

        # mention annotation
        positive_mentions = [[item for sublist in x['entities_tokenized'] for item in sublist] for x in batch]
        negative_mentions = [x['negative_mentions'] for x in batch]
        entities = [x['entities'] for x in batch]
        all_mentions = [a+b for a,b in zip(positive_mentions, negative_mentions)]

        return {
            'input_ids': input_ids,
            'labels': ner_tags_batch,
            'entities': positive_mentions,
            'label_mask': label_mask.unsqueeze(-1)
        }
    
    def collateExp5(self, batch):
        labelmaps = load_labelmaps(self.dataset_id)
        sequences = [torch.tensor(x['tokens']) for x in batch]
        masks = [torch.tensor([False]+([True]*(len(x)-1))) for x in sequences]
        ner_tags = []

        lens = [len(x) for x in sequences]
        
        for x in batch:
            nt = get_IOB_per_token(x, self.dataset_id)
            tags = [labelmaps["iob"][x][:2]+"SPAN" for x in nt]
            tags = [t.replace('OSPAN', 'O') for t in tags]
            ner_tags.append(tags)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) # pad token does not matter (autoregressive)
        label_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)

        # mention annotation
        positive_mentions = [[item[0] for sublist in x['entities_tokenized'] for item in sublist] for x in batch]
        positive_mentions_types = [[item[1] for sublist in x['entities'] for item in sublist] for x in batch]
        negative_mentions = [x['negative_mentions'] for x in batch]
        all_mentions = [a+b for a,b in zip(positive_mentions, negative_mentions)]

        # print(positive_mentions)
        return {
            'input_ids': input_ids,
            'mentions_types': positive_mentions_types,
            'mentions': positive_mentions,
            'lens': lens,
            'label_mask': label_mask.unsqueeze(-1),
            'iob': ner_tags,
            'tokens': sequences
        }   
     
    def collateExp6(self, batch):
        labelmaps = load_labelmaps(self.dataset_id)
        sequences = [torch.tensor(x['tokens']) for x in batch]
        masks = [torch.tensor([False]+([True]*(len(x)-1))) for x in sequences]
        ner_tags = []

        lens = [len(x) for x in sequences]
        
        for x in batch:
            nt = get_IOB_per_token(x, self.dataset_id)
            tags = [labelmaps["iob"][x] for x in nt]
            ner_tags.append(tags)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) # pad token does not matter (autoregressive)
        label_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)

        # mention annotation
        positive_mentions = [[item[0] for sublist in x['entities_tokenized'] for item in sublist] for x in batch]
        positive_mentions_types = [[item[1] for sublist in x['entities'] for item in sublist] for x in batch]
        negative_mentions = [x['negative_mentions'] for x in batch]
        all_mentions = [a+b for a,b in zip(positive_mentions, negative_mentions)]

        # print(positive_mentions)
        return {
            'input_ids': input_ids,
            'mentions_types': positive_mentions_types,
            'mentions': positive_mentions,
            'lens': lens,
            'label_mask': label_mask.unsqueeze(-1),
            'iob': ner_tags,
            'tokens': sequences
        }
    
def collateExp9(batch):
        sequences = [torch.tensor(x['full']) for x in batch]
        start_of_generated = [len(x['prompt']) for x in batch]

        lens = [len(x) for x in sequences]

        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) # pad token does not matter (autoregressive)

        mentions = [x['mentions'] for x in batch]
        
        return {
            'input_ids': input_ids,
            'mentions': mentions,
            'lens': lens,
            'lens_prompt': start_of_generated,
            'tokens': sequences
        }   

def get_features_and_labels_exp1(model, loader, layer_id):
    features = []
    labels = []
    for batch in tqdm(loader):
    
        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
        h_s_at_layer = outputs.hidden_states[layer_id].detach().cpu()
        h_s_selected = torch.masked_select(h_s_at_layer, batch['label_mask']).view(-1, outputs.hidden_states[2].size(-1))
        labels_selected = torch.masked_select(batch['labels'], batch['label_mask'].squeeze(-1))
    
        features.append(h_s_selected)
        labels.append(labels_selected)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

def get_features_and_labels_exp4(model, loader):
    features = []
    labels = []
    for batch in tqdm(loader):
    
        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
        attentions = torch.stack(outputs.attentions).swapaxes(0,1)
        att_feature = torch.zeros((batch['labels'].size(0), batch['labels'].size(1),attentions.size(1), attentions.size(2)))
        #print(attentions.size(), att_feature.size())
        for i in range(2, batch['labels'].size(1)):
            #att_feature[:, i] = attentions[:, :, :, i, i-1]
            att_feature[:, i] = torch.softmax(attentions[:, :, :, i, max(1, i-5):i+1], dim=-1)[:, :, :, -2]
        h_s_at_layer = att_feature.detach().cpu().view(att_feature.size(0), att_feature.size(1), att_feature.size(-1)*att_feature.size(-2))
        h_s_selected = torch.masked_select(h_s_at_layer, batch['label_mask']).view(-1, att_feature.size(-1)*att_feature.size(-2))
        labels_selected = torch.masked_select(batch['labels'], batch['label_mask'].squeeze(-1))
    
        features.append(h_s_selected)
        labels.append(labels_selected)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

def create_mask_for_len(seq_len, pad_to=None, skip_start=0, window=None):
    mask = (-1 * (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) - 1)).bool()

    if skip_start != 0:
        mask[:skip_start, :] = False
        mask[:, :skip_start] = False
    
    if window is not None:
        for i in range(window, seq_len):
            mask[i, :max(i-window, 0)] = False

    if pad_to is None:
        return mask
    
    return F.pad(mask, (0, pad_to-seq_len, 0, pad_to-seq_len))
def get_features_and_labels_exp5(model, loader):
    features = []
    labels = []

    for batch in tqdm(loader):

        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
        attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()

        attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)

        # create label tensors
        lb_tensor = torch.zeros((attentions.shape[0], attentions.shape[1], attentions.shape[2]))
        for i, sample in enumerate(batch['mentions']):
            for y, x in sample:
                lb_tensor[i,x,y] = 1
        
        # create mask
        mask = torch.stack([create_mask_for_len(x, max(batch['lens'])) for x in batch['lens']])
        

        labels_masked = torch.masked_select(lb_tensor, mask)
        attentions_masked = torch.masked_select(attentions, mask.unsqueeze(-1)).view(-1, attentions.size(-1))

        
        features.append(attentions_masked)
        labels.extend(labels_masked)

    features = torch.cat(features, dim=0)
    labels = torch.stack(labels)
    
    return features, labels

def get_features_and_labels_exp6(model, loader):
    features = []
    labels = []

    for batch in tqdm(loader):

        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
        attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()

        attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)

        # create label tensors
        lb_tensor = torch.zeros((attentions.shape[0], attentions.shape[1], attentions.shape[2]))
        for i, sample in enumerate(batch['mentions']):
            for y, x in sample:
                lb_tensor[i,x+1,y] = 1
        
        # create mask
        mask = torch.stack([create_mask_for_len(x, max(batch['lens'])) for x in batch['lens']])
        

        labels_masked = torch.masked_select(lb_tensor, mask)
        attentions_masked = torch.masked_select(attentions, mask.unsqueeze(-1)).view(-1, attentions.size(-1))
        
        features.append(attentions_masked)
        labels.extend(labels_masked)

    features = torch.cat(features, dim=0)
    labels = torch.stack(labels)
    
    return features, labels

def get_features_and_labels_exp7(model, loader):
    features = []
    labels = []

    for batch in tqdm(loader):

        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
        attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()

        attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)
        attentions_shifted = torch.cat((torch.zeros((attentions.size(0), 1, attentions.size(2), attentions.size(3))), attentions), dim=1)

        attentions = torch.cat((attentions_shifted[:, 1:, :, :], attentions), dim=-1)

        # create label tensors
        lb_tensor = torch.zeros((attentions.shape[0], attentions.shape[1], attentions.shape[2]))
        for i, sample in enumerate(batch['mentions']):
            for y, x in sample:
                lb_tensor[i,x+1,y] = 1
        
        # create mask
        mask = torch.stack([create_mask_for_len(x, max(batch['lens'])) for x in batch['lens']])
        mask[:, 0, :] = False
        labels_masked = torch.masked_select(lb_tensor, mask)
        attentions_masked = torch.masked_select(attentions, mask.unsqueeze(-1)).view(-1, attentions.size(-1))
        
        features.append(attentions_masked)
        labels.extend(labels_masked)

    features = torch.cat(features, dim=0)
    labels = torch.stack(labels)
    
    return features, labels

def get_features_for_all_layers_exp1(model, loader):

    features, labels = [], []
    
    for layer in range(1,13):
        features_l, labels_l = get_features_and_labels_exp1(model, loader, layer_id=layer)
        labels = labels_l
        features.append(features_l)

    features = torch.stack(features).swapaxes(0, 1)

    return features, labels

def get_features_and_labels_exp9(model, loader):
    features = []
    labels = []

    for batch in tqdm(loader):

        with torch.no_grad():
            #print(batch['input_ids'][:,:2048].size())
            outputs = model(batch['input_ids'][:,:2048].to(model.device), output_attentions=True)
            attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()
            del outputs
            torch.cuda.empty_cache()
    
            attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)
    
            # create label tensors
            lb_tensor = torch.zeros((attentions.shape[0], attentions.shape[1], attentions.shape[2]))
            max_len = min(max(batch['lens']), 2048)
            for i, sample in enumerate(batch['mentions']):
                for y, x in sample:
                    if x+1 < max_len:
                        lb_tensor[i,x+1,y] = 1
            
            # create mask
            mask = torch.stack([create_mask_for_len(x, max_len, skip_start=xp, window=15) for x, xp in zip([min(l, max_len) for l in batch['lens']], batch['lens_prompt'])])
    
            labels_masked = torch.masked_select(lb_tensor, mask)
            attentions_masked = torch.masked_select(attentions, mask.unsqueeze(-1)).view(-1, attentions.size(-1))
            
            features.append(attentions_masked)
            labels.extend(labels_masked)

    features = torch.cat(features, dim=0)
    labels = torch.stack(labels)
    
    return features, labels
