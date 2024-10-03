from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
# import datasets
import torch
# from transformers import pipeline
from torch.nn.parameter import Parameter
import copy
from torch.utils.data import DataLoader
from transformers import get_scheduler, AdamW
from tqdm.auto import tqdm
import torch.nn.functional as F
from datetime import datetime
from torch.optim import SGD
import os
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed(42)

SAVE_PATH = ...
CACHE_DIR = ...

currentDateAndTime = datetime.now()
save_path = os.path.join(SAVE_PATH)
os.makedirs(save_path)

tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb", cache_dir = CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb", CACHE_DIR)
eval_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb", CACHE_DIR)

embed = model.get_input_embeddings()

def filter_0(example):
    if example['label'] == 0:
        return True
    return False

def filter_1(example):
    if example['label'] == 1:
        return True
    return False

def emb2indices(output, emb_layer_weight):
    # output is size: [batch, sequence, emb_length], emb_layer is size: [num_tokens, emb_length]
    with torch.no_grad():
        emb_weights = emb_layer_weight

        # get indices from embeddings:
        emb_size = output.size(0), output.size(1), -1, -1
        out_size = -1, -1, emb_weights.size(0), -1
        residual = torch.abs(output.unsqueeze(2).expand(out_size) -
                                        emb_weights.unsqueeze(0).unsqueeze(0).expand(emb_size)).sum(dim=3)
        out_indices = torch.argmin(residual, dim=2)
        minimum_shift = torch.min(residual, dim=2)
        return out_indices, minimum_shift[0].squeeze().nonzero().reshape((1, -1))

def soft_threshold(delta, threshold):
    larger = delta > threshold
    smaller = delta < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    delta = delta * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    delta = delta + subtracted + added

    return delta

imdb = load_dataset("imdb", cache_dir = '/datasets')

def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True)
    return outputs

tokenized_ds = imdb.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

tokenized_ds0 = tokenized_ds.filter(filter_0)
tokenized_ds1 = tokenized_ds.filter(filter_1)

tokenized_datasets0 = tokenized_ds0.remove_columns(["text", "label"])
tokenized_datasets1 = tokenized_ds1.remove_columns(["text", "label"])

tokenized_datasets0.set_format("torch")
tokenized_datasets1.set_format("torch")

BATCHSIZE = 8

train_dataloader0 = DataLoader(
    tokenized_datasets0["train"], shuffle=True, batch_size=BATCHSIZE, collate_fn=data_collator
)
train_dataloader1 = DataLoader(
    tokenized_datasets1["train"], shuffle=True, batch_size=BATCHSIZE, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets0["test"], batch_size=1, collate_fn=data_collator
)

'''

    embeddings = embed(batch['input_ids'])
    print(embeddings.shape)

    logit = model(inputs_embeds=embeddings, attention_mask = batch['attention_mask'])

'''

def test(L1, L2, num_samples, inner_iter):
    CKPT_PATH = ...
    model.load_state_dict(torch.load(CKPT_PATH))

    test_evaluator = DataLoader(
        tokenized_datasets0["test"], batch_size=1, collate_fn=data_collator, shuffle=True
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    for i in range(num_samples):
        batch = {k: v.to(device) for k, v in next(iter(test_evaluator)).items()}

        embeddings = embed(batch['input_ids'])
        embeddings = Parameter(embeddings, requires_grad = True)

        delta = torch.zeros(embeddings.shape)
        delta = delta.cuda()
        delta = Parameter(delta, requires_grad = True)

        eps = 1e-8
        for j in range(inner_iter):

            logit = model(inputs_embeds=embeddings+delta, attention_mask = batch['attention_mask']).logits
            activations = torch.sigmoid(logit)

            # loss_objective = torch.log(1 - activations + eps).mean() 
            loss_objective = torch.log(1 - activations[0, 1] + eps).mean() 
            loss_penalty = L2 * (torch.norm(delta, p=2, dim=(1, 2)) ** 2).mean()

            loss = loss_objective + loss_penalty

            gradient = torch.autograd.grad(loss, delta)[0]
            stepsize = 1
            new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1, 2), keepdim=True))
            if L1>0:
                new_delta = soft_threshold(new_delta, L1)

            delta = new_delta

        id, nonzeros = emb2indices((embeddings+delta)[[0]].cpu(), embed.weight.cpu())
        print("****   Original:   ****", tokenizer.batch_decode(batch['input_ids'][[0]]))

        mask = torch.zeros_like(batch['input_ids'][[0]])    
        for idx in nonzeros[0]:
            mask[0, idx]=1

        print("****   New:        ****", tokenizer.batch_decode(mask * batch['input_ids'][[0]]))

def train(L1, L2, num_epochs, train_delta_itr):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader0)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    embed.requires_grad_(False)
    eval_model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    eval_model.eval()
    num_itr = 0
    eps = 1e-8

    for epoch in range(num_epochs):
        for batch in train_dataloader0:
            num_itr += 1

            batch = {k: v.to(device) for k, v in batch.items()}

            embeddings = embed(batch['input_ids'])
            embeddings = Parameter(embeddings, requires_grad = True)
            
            delta = torch.zeros(embeddings.shape)
            delta = delta.cuda()
            delta = Parameter(delta, requires_grad = True)

            for j in range(train_delta_itr):

                logit = model(inputs_embeds=embeddings+delta, attention_mask = batch['attention_mask']).logits
                activations = torch.sigmoid(logit)

                loss_objective = torch.log(1 - activations + eps).mean() 
                loss_penalty = L2 * (torch.norm(delta, p=2, dim=(1, 2)) ** 2).mean()
                if L1>0:
                    loss_penalty += L1 * (torch.norm(delta, p=1, dim=(1, 2))).mean()
                loss = loss_objective + loss_penalty

                gradient = torch.autograd.grad(loss, delta)[0]
                stepsize = 10
                new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1, 2), keepdim=True))
                delta = new_delta


            batch1 = next(iter(train_dataloader1))
            batch1 = {k: v.to(device) for k, v in batch1.items()}

            fake_prediction = torch.sigmoid(model(inputs_embeds=embeddings+delta, attention_mask = batch['attention_mask']).logits)
            real_prediction = torch.sigmoid(model(**batch1).logits)
            wass_loss = -1 * (torch.log(real_prediction + eps).mean() + torch.log(1 - fake_prediction + eps).mean())

            acc = ((fake_prediction < 0.5).sum() + (real_prediction > 0.5).sum()) / (BATCHSIZE * 2)

            wass_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_description("Norm2:{:.4f}, Norm1: {:.4f}, Loss_D:{:.4f}, Loss_trans:{:.4f}, Acc:{:.4f}".format(torch.norm(delta, p=2, dim=(1, 2)).mean(), torch.norm(delta, p=1, dim=(1, 2)).mean(), 
                                                                    -1 * wass_loss.item(), loss_objective.item(), acc.item()))
            progress_bar.update(1)

            if num_itr % 100 == 0:
                print("****   Original:   ****", tokenizer.batch_decode(batch['input_ids'][[0]]))
                print("****   New:        ****", tokenizer.batch_decode(emb2indices((embeddings+delta)[[0]].cpu(), embed.weight.cpu())[0]))

                eval_max_itr = 10
                success_rate = 0
                for i, eval_batch in enumerate(eval_dataloader):
                    if i == eval_max_itr:
                        break

                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    embeddings = embed(eval_batch['input_ids'])

                    delta = torch.zeros(embeddings.shape)
                    delta = delta.cuda()
                    delta = Parameter(delta, requires_grad = True)

                    for j in range(train_delta_itr):

                        logit = model(inputs_embeds=embeddings+delta, attention_mask = eval_batch['attention_mask']).logits
                        activations = torch.sigmoid(logit)

                        loss_objective = torch.log(1 - activations).mean() 
                        loss_penalty = L2 * (torch.norm(delta, p=2, dim=(1, 2)) ** 2).mean()
                        if L1>0:
                            loss_penalty += L1 * (torch.norm(delta, p=1, dim=(1, 2))).mean()
                        loss = loss_objective + loss_penalty

                        gradient = torch.autograd.grad(loss, delta)[0]
                        stepsize = 1
                        new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1, 2), keepdim=True))
                        delta = new_delta

                    pred = eval_model(inputs_embeds=embeddings+delta, attention_mask = eval_batch['attention_mask']).logits
                    success_rate += (pred > 0.5).sum() / len(pred)

                print("Success rate: {:.4f}".format(success_rate / eval_max_itr))

                torch.save(model.state_dict(), os.path.join(save_path, '{}_suc{:.4f}_2norm{:.4f}_1norm{:.4f}.pth'.format(num_itr, success_rate / eval_max_itr, 
                                                                                                                    torch.norm(delta, p=2, dim=(1, 2)).mean(), torch.norm(delta, p=1, dim=(1, 2)).mean())))

def eval(model_path, plot_color, eval_max_itr, inner_max_itr, L1, L2, p=2):

    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    embed.requires_grad_(False)
    eval_model.to(device)

    model.train()
    eval_model.eval()

    eval_max_itr = 400
    avg = 0
    xs = []
    ys = []

    for i, eval_batch in tqdm(enumerate(eval_dataloader), total=eval_max_itr):
        if i == eval_max_itr:
            break

        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        embeddings = embed(eval_batch['input_ids'])

        delta = torch.zeros(embeddings.shape)
        delta = delta.cuda()
        delta = Parameter(delta, requires_grad = True)

        for j in range(inner_max_itr):

            logit = model(inputs_embeds=embeddings+delta, attention_mask = eval_batch['attention_mask']).logits
            activations = torch.sigmoid(logit)

            loss_objective = torch.log(1 - activations).mean() 
            loss_penalty = L2 * (torch.norm(delta, p=2, dim=(1, 2)) ** 2).mean()
            
            loss = loss_objective + loss_penalty

            gradient = torch.autograd.grad(loss, delta)[0]
            stepsize = 1
            new_delta = delta - stepsize * (gradient / torch.norm(gradient, p=2, dim=(1, 2), keepdim=True))
            if L1>0:
                new_delta = soft_threshold(new_delta, L1)
            delta = new_delta

        pred = eval_model(inputs_embeds=embeddings+delta, attention_mask = eval_batch['attention_mask']).logits
        pred = torch.softmax(pred, dim=1)
        avg += pred[:, 1].detach().cpu().numpy()
        x = (torch.norm(delta, p=p, dim=(1, 2)) / (delta.shape[1] * delta.shape[2])).detach().cpu().numpy()
        y = pred[:, 1].detach().cpu().numpy()
        xs.append(x)
        ys.append(y)

    save_dir = f'./npys{eval_max_itr}/{L1}'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'data.npy'), 'wb') as f:
        np.save(f, xs)
        np.save(f, ys)
    f.close()

    with open(os.path.join(save_dir, 'data.npy'), 'rb') as f:
        xs = np.load(f)
        ys = np.load(f)

    plt.scatter(xs, ys, color=plot_color)

    avg /= eval_max_itr
    print('avg: ', L1, avg)
    plt.plot([0., 1.], [avg, avg], color = plot_color)