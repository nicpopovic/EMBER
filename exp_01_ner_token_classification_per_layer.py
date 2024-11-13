# ----- imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import MulticlassF1Score

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.data.loaders import *
from src.util import get_iob_labels, print_metric, get_class_labels
from src.classifier.ner import MLP
from src.data.ner import get_IOB_per_token, CollateManagerExp1, get_features_and_labels_exp1
from src.data.llama3tokenizer import CustomLlama3Tokenizer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

import wandb
import string
import random
import argparse

# ----- argument parsing and wandb init
random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2", help="model")
parser.add_argument("--dataset", type=str, default="conll2003", help="dataset")
parser.add_argument("--data_dir", type=str, default="featurecache/exp_01", help="dataset")
parser.add_argument("--batch_size_train", type=int, default=512, help="training batch size")
parser.add_argument("--batch_size_gen", type=int, default=4, help="feature generation batch size")
parser.add_argument("--classifier_hidden_dim", type=int, default=32, help="MLP hidden layer size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="classifier learning rate")
parser.add_argument("--num_epochs", type=int, default=25, help="number of training epochs")
parser.add_argument("--random_seed", type=int, default=1, help="random seed")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/exp_01", help="checkpoint_dir")
parser.add_argument("--project_name", type=str, default="2311_embedIE_01_NER_token_classification_per_layer", help="project_name")
parser.add_argument("--feature_layer", type=int, default=1, help="layer to extract NER feature from")

parser.add_argument('--dev', action='store_true')
parser.add_argument('--no-dev', dest='dev', action='store_false')
parser.set_defaults(dev=False)

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)

parser.add_argument("--hfcache", type=str, default="/pfs/work7/workspace/scratch/jg2894-ws_aug/cache/", help="huggingface cache")

args = parser.parse_args()

wandb.init(project=args.project_name+"_"+args.dataset.split("/")[-1])
wandb.config.update(args)
wandb.config.identifier = random_string

os.environ["TOKENIZERS_PARALLELISM"] = "false"

kwds = {
    "token": os.getenv("HF_TOKEN")
}

if args.hfcache != "":
    kwds["cache_dir"] = args.hfcache


# ----- model & data loading

model_name = args.model
dataset_id = args.dataset
batch_size = args.batch_size_gen


datasets = {}

model = None

for split in ["train", "validation", "test"]:


    dev_string = ""
    if args.dev:
        dev_string = "_dev"

    folder_name = f"{model_name.replace('/', '-')}_{dataset_id.replace('/', '_')}_{split}_{args.feature_layer}{dev_string}"

    
    try:

        # LOAD NER DATA
        features_gen = torch.load(os.path.join(args.data_dir, os.path.join(folder_name, "ft.pt"))).to(torch.float32)
        labels_gen = torch.load(os.path.join(args.data_dir, os.path.join(folder_name, "lb.pt"))).long()

        datasets[split] = TensorDataset(features_gen, labels_gen)


        print(f"loaded {split} split from cache")
    except:
        print(f"failed to load {split} split from cache, generating attention features and saving to '{os.path.join(args.data_dir, folder_name)}'")


        if model is None:
            if "llama" in model_name.lower():
                tokenizer = CustomLlama3Tokenizer(model_name, use_fast=True, **kwds)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwds)
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=False, output_hidden_states=True, return_dict=True, device_map="auto", **kwds).half()

        dataset_gen = FeatureGenDataset(dataset_id, split, dev=args.dev, tokenizer=tokenizer, force_bos=True, rel=False, coref=False)

        dataloader_gen = DataLoader(dataset_gen, batch_size=batch_size, shuffle=False, collate_fn=CollateManagerExp1(dataset_id).collateExp1)


        # ----- create features and labels

        features_gen, labels_gen = get_features_and_labels_exp1(model, dataloader_gen, layer_id=args.feature_layer)

        os.makedirs(os.path.join(args.data_dir, folder_name), exist_ok=True)
        torch.save(features_gen, os.path.join(args.data_dir, folder_name, "ft.pt"))
        torch.save(labels_gen, os.path.join(args.data_dir, folder_name, "lb.pt"))

        datasets[split] = TensorDataset(features_gen.to(torch.float32), labels_gen.long())

# free up gpu memory
model = None

batch_size = args.batch_size_train

dataloader_train = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=20)
dataloader_validation = DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=20)
dataloader_test = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=20)


# ----- setup classifier and optimizer

num_epochs=args.num_epochs

classifier = MLP(datasets["train"][0][0].size(-1), len(get_iob_labels(dataset_id)), hidden_dim=args.classifier_hidden_dim, cuda=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=args.learning_rate, eps=1e-6)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, len(dataloader_train), num_epochs*len(dataloader_train))

# ----- training loop

best_f1 = 0
is_best = False
step_global = -1

def iob_to_classwise(preds, labels, dataset_id, first_only=False):
    map = []

    new_tags = sorted(set(["-".join(x.split("-")[1:]) for x in get_iob_labels(dataset_id)]), key=["-".join(x.split("-")[1:]) for x in get_iob_labels(dataset_id)].index)

    tags = get_iob_labels(dataset_id)
    for i, tag in enumerate(tags):
        if tag.startswith("I-") or tag.startswith("B-"):
            map.append(new_tags.index(tag[2:]))
        else:
            map.append(i)

    labels_new = torch.zeros_like(torch.Tensor(labels))
    preds_new = torch.zeros_like(torch.Tensor(preds))

    for i, j in enumerate(map):
        labels_new[labels == i] = j
        preds_new[preds == i] = j

    return preds_new, labels_new



for epoch in range(num_epochs):

    # TRAIN
    with tqdm(dataloader_train, f"training epoch {epoch+1}") as prog:

        losses = []
        metric = MulticlassF1Score(num_classes=len(get_iob_labels(dataset_id)), average=None, device=classifier.device)
        classifier.train()
        for item in prog:
            pred_ner = classifier(item[0].cuda())
            labels_ner = item[1].cuda()

            loss = criterion(pred_ner, labels_ner)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.append(loss.item())

            """
            if step_global % 100 == 0:
                wandb.log({"loss": sum(losses)/len(losses)}, step=step_global)
            """

            metric.update(pred_ner, labels_ner)

            step_global += 1

        print("NAMED ENTITY RECOGNITION - TRAIN - IOB")
        p_micro, r_micro, f_micro = print_metric(metric, get_iob_labels(dataset_id))

        wandb.log({"precision_train": p_micro}, step=step_global)
        wandb.log({"recall_train": r_micro}, step=step_global)
        wandb.log({"f1_train": f_micro}, step=step_global)

    # EVAL
    with torch.no_grad():
        classifier.eval()

        for loader, loadername in zip([dataloader_validation, dataloader_test], ["validation", "test"]):

            if loadername == "test" and not is_best:
                continue

            is_best = False

            preds = []
            labels = []

            metric_iob = MulticlassF1Score(num_classes=len(get_iob_labels(dataset_id)), average=None, device=classifier.device)
            metric_classwise = MulticlassF1Score(num_classes=len(get_class_labels(dataset_id)), average=None, device=classifier.device)

            with tqdm(loader) as prog:

                for item in prog:
                    pred_ner = classifier(item[0].cuda())

                    labels_ner = item[1].cuda()

                    # save for confusion matrix
                    if loadername == "test":
                        preds.append(pred_ner.detach().cpu())
                        labels.append(labels_ner.detach().cpu())

                    # Eval NER
                    metric_iob.update(pred_ner, labels_ner)
                    metric_classwise.update(*iob_to_classwise(torch.argmax(pred_ner, dim=-1), labels_ner, dataset_id))

            print(f"NAMED ENTITY RECOGNITION - {loadername} - IOB")
            (p_micro, r_micro, f_micro), metrics = print_metric(metric_iob, get_iob_labels(dataset_id), return_classwise=True)


            wandb.log({f"precision_{loadername}_IOB": p_micro}, step=step_global)
            wandb.log({f"recall_{loadername}_IOB": r_micro}, step=step_global)
            wandb.log({f"f1_{loadername}_IOB": f_micro}, step=step_global)


            for class_label in metrics.keys():
                wandb.log({f"precision_{loadername}_{class_label}_iob": metrics[class_label]["p"]}, step=step_global)
                wandb.log({f"recall_{loadername}_{class_label}_iob": metrics[class_label]["r"]}, step=step_global)
                wandb.log({f"f1_{loadername}_{class_label}_iob": metrics[class_label]["f"]}, step=step_global)


            print(f"NAMED ENTITY RECOGNITION - {loadername} - CLASSWISE")
            (p_micro, r_micro, f_micro), metrics = print_metric(metric_classwise, sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset_id)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset_id)].index), return_classwise=True)

            wandb.log({f"precision_{loadername}_CLASSWISE": p_micro}, step=step_global)
            wandb.log({f"recall_{loadername}_CLASSWISE": r_micro}, step=step_global)
            wandb.log({f"f1_{loadername}_CLASSWISE": f_micro}, step=step_global)


            for class_label in metrics.keys():
                wandb.log({f"precision_{loadername}_{class_label}": metrics[class_label]["p"]}, step=step_global)
                wandb.log({f"recall_{loadername}_{class_label}": metrics[class_label]["r"]}, step=step_global)
                wandb.log({f"f1_{loadername}_{class_label}": metrics[class_label]["f"]}, step=step_global)


            if loadername == "test":
                # confusion matrix
                preds = torch.vstack(preds).max(dim=-1).indices.numpy()
                labels = torch.cat(labels).numpy()
                cm_iob = confusion_matrix(labels, preds, normalize='true')
                cm_classwise = confusion_matrix(*iob_to_classwise(labels, preds, dataset_id), normalize='true')
                lb_cw, pr_cw = iob_to_classwise(labels, preds, dataset_id)

                """
                wandb.log({"conf_mat_iob" : wandb.plot.confusion_matrix(probs=None,
                        y_true=labels, preds=preds,
                        class_names=get_iob_labels(dataset_id))})

                wandb.log({"conf_mat_classwise" : wandb.plot.confusion_matrix(probs=None,
                                        y_true=lb_cw.numpy(), preds=pr_cw.numpy(),
                                        class_names=sorted(set([x.split("-")[-1] for x in get_iob_labels(dataset_id)]), key=[x.split("-")[-1] for x in get_iob_labels(dataset_id)].index))})
                """
                torch.save(torch.Tensor(cm_iob), os.path.join(args.checkpoint_dir, f"cm_iob_{random_string}.pt"))
                torch.save(torch.Tensor(cm_classwise), os.path.join(args.checkpoint_dir, f"cm_classwise_{random_string}.pt"))

                #print(cm_iob, cm_classwise)
            elif f_micro > best_f1:
                best_f1 = f_micro
                is_best = True

                wandb.log({f"best_precision_{loadername}": p_micro}, step=step_global)
                wandb.log({f"best_recall_{loadername}": r_micro}, step=step_global)
                wandb.log({f"best_f1_{loadername}": f_micro}, step=step_global)
                os.makedirs(args.checkpoint_dir, exist_ok=True)

                torch.save(classifier.state_dict(), os.path.join(args.checkpoint_dir, f"checkpoint_{random_string}.pt"))

                print("weights saved!")
