# ----- imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import BinaryF1Score

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.data.loaders import *
from src.util import get_iob_labels, print_metric
from src.classifier.ner import MLP
from src.data.ner import get_IOB_per_token, CollateManagerExp1, get_features_and_labels_exp4

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
parser.add_argument("--data_dir", type=str, default="featurecache/exp_04", help="dataset")
parser.add_argument("--batch_size_train", type=int, default=128, help="training batch size")
parser.add_argument("--batch_size_gen", type=int, default=4, help="feature generation batch size")
parser.add_argument("--classifier_hidden_dim", type=int, default=32, help="MLP hidden layer size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="classifier learning rate")
parser.add_argument("--num_epochs", type=int, default=50, help="number of training epochs")
parser.add_argument("--random_seed", type=int, default=1, help="random seed")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/exp_04", help="checkpoint_dir")
parser.add_argument("--project_name", type=str, default="2311_embedIE_04_NER_span_classification", help="project_name")

parser.add_argument('--dev', action='store_true')
parser.add_argument('--no-dev', dest='dev', action='store_false')
parser.set_defaults(dev=False)

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)

parser.add_argument("--hfcache", type=str, help="huggingface cache")

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

    folder_name = f"{model_name.replace('/', '-')}_{dataset_id.replace('/', '_')}_{split}{dev_string}"

    
    try:

        # LOAD NER DATA
        features_gen = torch.load(os.path.join(args.data_dir, os.path.join(folder_name, "ft.pt"))).to(torch.float32)
        labels_gen = torch.load(os.path.join(args.data_dir, os.path.join(folder_name, "lb.pt"))).long()

        datasets[split] = TensorDataset(features_gen, labels_gen)


        print(f"loaded {split} split from cache")
    except:
        print(f"failed to load {split} split from cache, generating attention features and saving to '{os.path.join(args.data_dir, folder_name)}'")


        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwds)
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True, return_dict=True, device_map="auto", **kwds).half()

        dataset_gen = FeatureGenDataset(dataset_id, split, dev=args.dev, tokenizer=tokenizer, force_bos=True, rel=False, coref=False)

        dataloader_gen = DataLoader(dataset_gen, batch_size=batch_size, shuffle=False, collate_fn=CollateManagerExp1(dataset_id).collateExp4)


        # ----- create features and labels

        features_gen, labels_gen = get_features_and_labels_exp4(model, dataloader_gen)

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

classifier = MLP(datasets["train"][0][0].size(-1), 2, hidden_dim=args.classifier_hidden_dim, cuda=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=args.learning_rate, eps=1e-6)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, len(dataloader_train), num_epochs*len(dataloader_train))

# ----- training loop

best_f1 = 0
is_best = False
step_global = -1



for epoch in range(num_epochs):

    # TRAIN
    with tqdm(dataloader_train, f"training epoch {epoch+1}") as prog:

        losses = []
        metric = BinaryF1Score()
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

            metric.update(torch.nn.functional.softmax(pred_ner, dim=-1)[:,1], labels_ner)

            step_global += 1

        f_micro = metric.compute()

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

            metric = BinaryF1Score()

            with tqdm(loader) as prog:

                for item in prog:
                    pred_ner = classifier(item[0].cuda())
                    labels_ner = item[1].cuda()

                    # Eval NER
                    metric.update(torch.nn.functional.softmax(pred_ner, dim=-1)[:,1], labels_ner)

            f_micro = metric.compute()

            wandb.log({f"f1_{loadername}": f_micro}, step=step_global)


            if loadername != "test" and f_micro > best_f1:
                best_f1 = f_micro
                is_best = True

                wandb.log({f"best_f1_{loadername}": f_micro}, step=step_global)
                os.makedirs(args.checkpoint_dir, exist_ok=True)

                torch.save(classifier.state_dict(), os.path.join(args.checkpoint_dir, f"checkpoint_{random_string}.pt"))

                print("weights saved!")
