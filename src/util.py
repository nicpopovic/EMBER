import torch


def print_metric(metric, class_labels, return_classwise=False):
    
    f_ner = metric.compute()
    p_ner = torch.nan_to_num(metric.num_tp / metric.num_prediction)
    r_ner = torch.nan_to_num(metric.num_tp / metric.num_label)

    print(f"{' '.ljust(10)}     P      R      F      S")

    sum_support = 0
    weighted_scores = [0, 0, 0]

    classwise = {}
    #print(f_ner)
    for ner_class, p, r, f, s in zip(class_labels, p_ner, r_ner, f_ner, metric.num_label):
        if ner_class == "NONE" or ner_class == "O" or ner_class == "no_relation":
            continue
        print(f"{ner_class.ljust(10)} - {p:.2f} - {r:.2f} - {f:.2f} - {int(s)}")
        weighted_scores[0] += p*s
        weighted_scores[1] += r*s
        weighted_scores[2] += f*s
        sum_support += s

        classwise[ner_class] = {"p": p, "r": r, "f": f}

    p_micro = weighted_scores[0]/sum_support
    r_micro = weighted_scores[1]/sum_support
    f_micro = weighted_scores[2]/sum_support

    classwise["macro"] = {"p": torch.mean(p_ner[1:]), "r": torch.mean(r_ner[1:]), "f": torch.mean(f_ner[1:])}

    print("")
    print(f"MICRO      - {p_micro:.2f} - {r_micro:.2f} - {f_micro:.2f}")
    print(f"MACRO      - {torch.mean(p_ner[1:]):.2f} - {torch.mean(r_ner[1:]):.2f} - {torch.mean(f_ner[1:]):.2f}")
    print("")

    if return_classwise:
        return (p_micro, r_micro, f_micro), classwise

    return p_micro, r_micro, f_micro

def get_class_labels(dataset):
    if dataset == "conll2003":
        return ["NONE", "PER", "ORG", "LOC", "MISC", "TIME", "NUM"]
    elif dataset == "docred":
        return ["NONE", "PER", "ORG", "LOC", "MISC"]
    elif dataset == "tner/ontonotes5":
        return ['NONE', 'CARDINAL', 'DATE', 'PERSON', 'NORP', 'GPE', 'LAW', 'PERCENT', 'ORDINAL', 'MONEY', 'WORK_OF_ART', 'FAC', 'TIME', 'QUANTITY', 'PRODUCT', 'LANGUAGE', 'ORG', 'LOC', 'EVENT']
    elif dataset == "tner/wnut2017":
        return ['NONE', 'corporation', 'creative-work', 'group', 'location', 'person', 'product']    
    elif dataset == "tner/multinerd":
        return ['NONE', 'PER','LOC','ORG','ANIM','BIO','CEL','DIS','EVE','FOOD','INST','MEDIA','PLANT','MYTH','TIME','VEHI','SUPER','PHY']
    elif dataset == "tner/bc5cdr":
        return ['NONE', 'Chemical', 'Disease']
    elif dataset == "retacred":
        return ['no_relation', 'org:alternate_names', 'org:city_of_branch', 'org:country_of_branch', 'org:dissolved', 'org:founded', 'org:founded_by', 'org:member_of', 'org:members', 'org:number_of_employees/members', 'org:political/religious_affiliation', 'org:shareholders', 'org:stateorprovince_of_branch', 'org:top_members/employees', 'org:website', 'per:age', 'per:cause_of_death', 'per:charges', 'per:children', 'per:cities_of_residence', 'per:city_of_birth', 'per:city_of_death', 'per:countries_of_residence', 'per:country_of_birth', 'per:country_of_death', 'per:date_of_birth', 'per:date_of_death', 'per:employee_of', 'per:identity', 'per:origin', 'per:other_family', 'per:parents', 'per:religion', 'per:schools_attended', 'per:siblings', 'per:spouse', 'per:stateorprovince_of_birth', 'per:stateorprovince_of_death', 'per:stateorprovinces_of_residence', 'per:title']
    else:
        raise NotImplementedError
    
def get_iob_labels(dataset):
    if dataset == "conll2003":
        return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    elif dataset == "tner/ontonotes5":
        return ['O', 'B-CARDINAL', 'B-DATE', 'I-DATE', 'B-PERSON', 'I-PERSON', 'B-NORP', 'B-GPE', 'I-GPE', 'B-LAW', 'I-LAW', 'B-ORG', 'I-ORG', 'B-PERCENT', 'I-PERCENT', 'B-ORDINAL', 'B-MONEY', 'I-MONEY', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-FAC', 'B-TIME', 'I-CARDINAL', 'B-LOC', 'B-QUANTITY', 'I-QUANTITY', 'I-NORP', 'I-LOC', 'B-PRODUCT', 'I-TIME', 'B-EVENT', 'I-EVENT', 'I-FAC', 'B-LANGUAGE', 'I-PRODUCT', 'I-ORDINAL', 'I-LANGUAGE']
    elif dataset == "tner/wnut2017":
        return ["O", "B-corporation","B-creative-work","B-group","B-location","B-person","B-product","I-corporation","I-creative-work","I-group","I-location","I-person","I-product"]
    elif dataset == "tner/multinerd":
        return ['O', "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL", "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD", "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-PLANT", "I-PLANT", "B-MYTH", "I-MYTH", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI", "B-SUPER", "I-SUPER", "B-PHY", "I-PHY"]
    elif dataset == "tner/bc5cdr":
        return ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
    else:
        raise NotImplementedError
