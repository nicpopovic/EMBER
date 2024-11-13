# ----- imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import BinaryF1Score

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.data.loaders import *
from src.util import get_iob_labels, print_metric
from src.classifier.ner import MLPexp05
from src.data.ner import get_IOB_per_token, CollateManagerExp1, get_features_and_labels_exp9, collateExp9
from src.data.llama3tokenizer import CustomLlama3Tokenizer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from seqeval.metrics import f1_score

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
parser.add_argument("--data_dir", type=str, default="featurecache/exp_09", help="dataset")
parser.add_argument("--batch_size_train", type=int, default=128, help="training batch size")
parser.add_argument("--batch_size_gen", type=int, default=4, help="feature generation batch size")
parser.add_argument("--classifier_hidden_dim", type=int, default=32, help="MLP hidden layer size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="classifier learning rate")
parser.add_argument("--num_epochs", type=int, default=25, help="number of training epochs")
parser.add_argument("--random_seed", type=int, default=1, help="random seed")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/exp_09", help="checkpoint_dir")
parser.add_argument("--project_name", type=str, default="2311_embedIE_09_NER_span_classification_next_generated", help="project_name")

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

model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True, return_dict=True, device_map="auto", **kwds).half()
if "llama" in model_name.lower():
    tokenizer = CustomLlama3Tokenizer(model_name, use_fast=True, **kwds)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwds)
  
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


        dataset_gen = JSONDataset(f"data_generated/conll2003_generated_{split}_final.json")

        dataloader_gen = DataLoader(dataset_gen, batch_size=batch_size, shuffle=False, collate_fn=collateExp9)


        # ----- create features and labels

        features_gen, labels_gen = get_features_and_labels_exp9(model, dataloader_gen)

        os.makedirs(os.path.join(args.data_dir, folder_name), exist_ok=True)
        torch.save(features_gen, os.path.join(args.data_dir, folder_name, "ft.pt"))
        torch.save(labels_gen, os.path.join(args.data_dir, folder_name, "lb.pt"))

        datasets[split] = TensorDataset(features_gen.to(torch.float32), labels_gen.long())


batch_size = args.batch_size_train

dataloader_train = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=20)
dataloader_validation = DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=20)
dataloader_test = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=20)

dataloader_test_seqeval = DataLoader(FeatureGenDataset(dataset_id, "test", dev=args.dev, tokenizer=tokenizer, force_bos=True, rel=False, coref=False), batch_size=args.batch_size_gen, shuffle=False, collate_fn=CollateManagerExp1(dataset_id).collateExp5)


# ----- setup classifier and optimizer

num_epochs=args.num_epochs

classifier = MLPexp05(datasets["train"][0][0].size(-1), 2, hidden_dim=args.classifier_hidden_dim, cuda=True)
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
            # -----------------------------

            if loadername == "test":
                true_seq = []
                pred_seq = []
                tokens = []
                for batch in tqdm(dataloader_test_seqeval):

                    with torch.no_grad():
                        true_seq.extend(batch['iob'])
                        
                        tokens.extend(batch['tokens'])
                        outputs = model(batch['input_ids'].to(model.device), output_attentions=True)
                        attentions = torch.stack(outputs.attentions).swapaxes(0,1).detach().cpu()
                    
                        attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)
                        
                        pred_ner = classifier(attentions.to(classifier.device).to(classifier.fc1.weight.dtype))

                    masked_predictions_argmax = torch.argmax(pred_ner, dim=-1)

                    score = (torch.argmax(pred_ner, dim=-1) * (pred_ner[:,:,:,1] - pred_ner[:,:,:,0]))

                    #print(masked_predictions_argmax[0])
                    indices_above_zero = torch.nonzero(score > 0)

                    values_above_zero = score[indices_above_zero[:, 0], indices_above_zero[:, 1], indices_above_zero[:, 2]]

                    if values_above_zero.shape[0] == 0:
                        iob_labels = [['O']*seq_len for seq_len in batch['lens']]
                        pred_seq.extend(iob_labels)
                        continue

                    # sort spans randomly breaking ties
                    vals, indices = zip(*sorted(zip(values_above_zero, indices_above_zero), reverse=True, key=lambda x: (x[0], random.random())))
                    
                    assigned = torch.zeros((score.shape[0], score.shape[1]))
                    
                    iob_labels = [['O']*seq_len for seq_len in batch['lens']]
                    
                    for ind in indices:

                        index = [*ind]
                        

                        if index[1] >= batch['lens'][index[0]]:
                            continue
                                    
                        check = assigned[index[0], index[2]:index[1]].sum()
                        if check > 0 or index[1] >= batch['lens'][index[0]]:
                            continue
                        assigned[index[0], index[2]:index[1]] = 1
                        iob_labels[index[0]][index[2]:index[1]] = ["I-SPAN"] * (index[1]-index[2])
                        iob_labels[index[0]][index[2]] = "B-SPAN"
                    pred_seq.extend(iob_labels)

                f1_seqeval = f1_score(true_seq, pred_seq)
                wandb.log({f"f1_{loadername}_seqeval": f1_seqeval}, step=step_global)

                # -----------------------------
