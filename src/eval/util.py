import torch
from src.data.ner import get_iob_labels


def iob_to_classwise(preds, dataset_id, first_only=False):
    map = []

    new_tags = sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset_id)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset_id)].index)

    tags = get_iob_labels(dataset_id)
    for i, tag in enumerate(tags):
        if tag.startswith("I-") or tag.startswith("B-"):
            map.append(new_tags.index(tag[2:]))
        else:
            map.append(i)

    preds_new = torch.zeros_like(torch.Tensor(preds))

    for i, j in enumerate(map):
        preds_new[preds == i] = j

    return preds_new


def to_mentions(iob_labels):

    output = []
    for x in iob_labels:
        if len(x) == 1:
            output.append(x)
        else:
            output.append(x[0] + "-MENTION")
    return output

def fuse_predictions(tokenwise, constituency, dataset_id):

    fused = torch.zeros_like(tokenwise)

    for batch_index in range(fused.size(0)):
        active_class = None
        for i in reversed(range(fused.size(-1))):
            if active_class is not None:
                # there is an active span
                # check if this is the first token of the span
                if constituency[batch_index, i] == 0:
                    # set first token and reset active class
                    fused[batch_index, i] = active_class[0]
                    active_class = None
                    continue
                else:
                    # we are in the middle of a span
                    fused[batch_index, i] = active_class[1]
                    continue
            else:
                # there is no currently active span
                # get token index for prediction
                tokens = iob_lookup(tokenwise[batch_index, i], dataset_id)
                # check if there is a span or if this is just a single token entity
                if constituency[batch_index, i] == 0 or tokens[0] == 0: #  or tokens[0] == 0
                    # single token entity
                    fused[batch_index, i] = tokens[0]
                    continue
                else:
                    # multi token entity
                    fused[batch_index, i] = tokens[1]
                    active_class = tokens
                    continue
                    
    # convert to strings
    fused_strings = []
    for j in range(fused.size(0)):
        seq = []
        for i in range(fused.size(-1)):
            seq.append(f"{get_iob_labels(dataset_id)[fused[j][i]]}")
        fused_strings.append(seq)
    
    return fused_strings

def fuse_predictions_2(tokenwise, span_predictions, dataset_id, force_span=False, conditional_force=False):

    fused = torch.zeros_like(tokenwise)

    for batch_index in range(fused.size(0)):
        active_class = None
        current_span_start = -1
        for i in reversed(range(fused.size(-1)-1)):
            
            if active_class is not None:
                # there is an active span
                # check if this is the first token of the span
                if current_span_start == i:
                    # set first token and reset active class
                    fused[batch_index, i] = active_class[0]
                    active_class = None
                    continue
                else:
                    # we are in the middle of a span
                    fused[batch_index, i] = active_class[1]
                    continue
            else:
                # there is no currently active span
                # get token index for prediction
                tokens = iob_lookup(tokenwise[batch_index, i], dataset_id)

                # most likely span that ends with token at i
                start_token_probabilities = span_predictions[batch_index, i+1, :, 1] - span_predictions[batch_index, i+1, :, 0]
                most_likely_start = torch.argmax(start_token_probabilities)
                #print(most_likely_start, start_token_probabilities)
            
                if torch.max(start_token_probabilities) < 0 and not force_span:
                    fused[batch_index, i] = 0
                    continue

                if torch.max(start_token_probabilities) < 0 and most_likely_start == i and conditional_force:
                    fused[batch_index, i] = 0
                    continue
                
                # check if there is a span or if this is just a single token entity
                if most_likely_start == i or tokens[0] == 0: #  or tokens[0] == 0
                    # single token entity
                    fused[batch_index, i] = tokens[0]
                    continue
                else:
                    # multi token entity
                    fused[batch_index, i] = tokens[1]
                    current_span_start = most_likely_start
                    active_class = tokens
                    continue

                    
    # convert to strings
    fused_strings = []
    for j in range(fused.size(0)):
        seq = []
        for i in range(fused.size(-1)):
            seq.append(f"{get_iob_labels(dataset_id)[fused[j][i]]}")
        fused_strings.append(seq)
    
    return fused_strings


def iob_lookup(token, dataset_id):
    if token == 0:
        return (0,0)

    # check token class
    prefix, class_label = get_iob_labels(dataset_id)[token].split("-")[0], "-".join(get_iob_labels(dataset_id)[token].split("-")[1:])

    b_token = get_iob_labels(dataset_id).index("B-"+class_label)
    i_token = get_iob_labels(dataset_id).index("I-"+class_label)
    return (b_token, i_token)
