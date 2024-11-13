# imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.loaders import *
from src.data.ner import CollateManagerExp1
from src.util import get_iob_labels
from src.classifier.ner import MLP
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2", help="model")
parser.add_argument("--checkpoint", type=str, default="checkpoints/exp_01/checkpoint_G3tzPf1hIC.pt", help="checkpoint_dir")
parser.add_argument("--classifier_hidden_dim", type=int, default=4096, help="MLP hidden layer size")
parser.add_argument("--hfcache", type=str, default="/pfs/work7/workspace/scratch/jg2894-ws_aug/cache/", help="huggingface cache")
parser.add_argument("--dataset", type=str, default="conll2003", help="dataset")
parser.add_argument("--feature_layer", type=int, default=10, help="layer to extract NER feature from")
parser.add_argument("--batch_size", type=int, default=1, help="feature generation batch size")
parser.add_argument('--dev', action='store_true')
parser.add_argument('--no-dev', dest='dev', action='store_false')
parser.set_defaults(dev=False)

args = parser.parse_args()

# load dataset on per sentence basis and evaluate model accordingly

# load model
model_name = args.model
kwds = {
    "token": os.getenv("HF_TOKEN")
}

if args.hfcache != "":
    kwds["cache_dir"] = args.hfcache

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwds)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=False, output_hidden_states=True, return_dict=True, device_map="auto", **kwds)


# load dataset
dataset_id = args.dataset
split = "validation"
batch_size = args.batch_size

dataset_gen = FeatureGenDataset(dataset_id, split, dev=args.dev, tokenizer=tokenizer, force_bos=True, rel=False, coref=False)
dataloader_gen = DataLoader(dataset_gen, batch_size=batch_size, shuffle=False, collate_fn=CollateManagerExp1(dataset_id).collateExp2)

# load classifier
classifier = MLP(model.config.n_embd, len(get_iob_labels(dataset_id)), hidden_dim=args.classifier_hidden_dim, cuda=True)
classifier.load_state_dict(torch.load(args.checkpoint, map_location=model.device))


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

print(get_iob_labels(dataset_id))
print(sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset_id)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset_id)].index))

# iterate over data samples and collect recall per span
def get_features_and_labels_exp1(model, loader, layer_id):
    predictions = []
    labels = []
    for batch in tqdm(loader):
        #print(batch['entities'], batch['input_ids'].size())
        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
        h_s_at_layer = outputs.hidden_states[layer_id]

        for ents, features in zip(batch['entities'], h_s_at_layer):
            pred_ner = classifier(features)
            pred_ner = iob_to_classwise(torch.argmax(pred_ner, dim=-1), dataset_id)
            #print(features.size(), pred_ner)
            for entity in ents:
                prediction_for_ent = pred_ner[entity[0][1]].cpu().numpy()

                #prediction_for_ent = pred_ner[entity[0][0]].cpu().numpy()

                #counts = np.bincount(pred_ner[entity[0][0]:entity[0][1]+1].cpu().numpy())
                #prediction_for_ent = np.argmax(counts)

                #counts = np.bincount(pred_ner[entity[0][0]:entity[0][1]+1].cpu().numpy())
                #counts[0] = 0
                #prediction_for_ent = np.argmax(counts)

                label = torch.argmax(entity[1], dim=-1)
                predictions.append(prediction_for_ent)
                labels.append(label)

    return predictions, labels

predictions, labels = get_features_and_labels_exp1(model, dataloader_gen, layer_id=args.feature_layer)

new_tags = sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset_id)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset_id)].index)

print(new_tags)

cm = confusion_matrix(labels, predictions, normalize='true')

disp = ConfusionMatrixDisplay(cm, display_labels=new_tags).plot()
fig, ax = plt.subplots(figsize=(20,20))
disp.plot(ax=ax)
plt.savefig(f'confusion_classes_{model_name.replace("/","-")}_{dataset_id.replace("/","-")}_{args.feature_layer}_{args.checkpoint.split("_")[-1].split(".")[0]}.png')

print(cm)