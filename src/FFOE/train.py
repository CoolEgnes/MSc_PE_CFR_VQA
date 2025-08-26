"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import torch
import src.utils as utils
import torch.nn as nn
from src.FFOE.trainer import Trainer
from lxrt.optimization import BertAdam
import pandas as pd
import pickle
import wandb
warmup_updates = 4000


def init_weights(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(m.weight)


def compute_score_with_logits(logits, labels, write=False):
    """
    20250703--zhd
    write=True, write the generated answers and ground truth answers into the csv to compare the results
    """
    # labels: Tensor: (256, 1533)
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)  # Tensor: (256,), the predicted label of each sample in the batch
    one_hots.scatter_(1, logits.view(-1, 1), 1)  # Tensor, (256, 1533)
    scores = (one_hots * labels)  # Tensor, (256, 1533)

    if write:
        return scores, logits
    return scores


def train(args, model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0, config=None):
    # with wandb.init(config=config) as run:
        # config = wandb.config
    # Initialize W&B if not already initialized
    # if not wandb.run:
    # wandb.init(project="CFR_VQA_sweep_test", config=args)
    # config1 = wandb.config
    # config2 = config
    
    print("==========================================")
    args.lxmert_lr = getattr(config, 'lxmert_lr', args.lxmert_lr)
    args.dropout = getattr(config, 'dropout', args.dropout)
    args.omega_v = getattr(config, 'omega_v', args.omega_v)

    print(f"lxmert_lr: {args.lxmert_lr}")
    print(f"dropout: {args.dropout}")
    print(f"omega_v: {args.omega_v}")

    device = args.device
    lr_default = args.lr  # 7e-4
    # lr_default =  getattr(wandb.config, 'lr_default', args.lr)

    run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="egnes-university-of-leeds",
            # Set the wandb project where this run will be logged.
            project="CFR_VQA_submit_local_test",
            name="submit_local_20250825",
            # Track hyperparameters and run metadata.
            config={"architecture": model}
        )

    lr_decay_step = 2
    # lr_decay_rate = .25
    lr_decay_rate = getattr(config, 'lr_decay_rate', 0.25)
    print(f"lr_decay_rate: {lr_decay_rate}")
    lr_decay_epochs = range(10, 20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]  # [0.00035, 0.0007, 0.00105, 0.0014]
    saving_epoch = 0
    grad_clip = args.clip_norm
    bert_optim = None

    # lxmert_lr = getattr(config, 'lxmert_lr', args.lxmert_lr)

    utils.create_dir(output)

    if args.model == 'CFRF_Model':
        batch_per_epoch = int(len(train_loader.dataset) / args.batch_size) + 1  # 14735 for train
        t_total = int(batch_per_epoch * args.epochs)  # 14735
        ignored_params = list(map(id, model.lxmert_encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        bert_optim = BertAdam(list(model.lxmert_encoder.parameters()),
                              lr=args.lxmert_lr,  # 1e-4
                              warmup=0.1,
                              t_total=t_total)

        optim = torch.optim.Adamax(list(base_params), lr=lr_default)

    else:
        raise BaseException("Model not found!")

    N = len(train_loader.dataset)  # 943000
    num_batches = int(N / args.batch_size + 1)  # 14735

    """
    torch.nn.BCEWithLogitsLoss: 
    This loss combines a Sigmoid layer and the BCELoss in one single class. 
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
    by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    BCELoss:
    Binary Cross Entropy between the target and the input probabilities
    """
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(args.__repr__())
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    trainer = Trainer(args, model, criterion, optim, bert_optim)
    update_freq = int(args.update_freq)  # 4
    wall_time_start = time.time()


    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        train_question_type_score = 0
        total_norm = 0
        count_norm = 0
        num_updates = 0
        t = time.time()
        if args.model == 'CFRF_Model':
            if epoch < len(gradual_warmup_steps):
                trainer.optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
                logger.write('gradual warmup lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
            elif epoch in lr_decay_epochs:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay_rate
                logger.write('decreased lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
            else:
                logger.write('lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
        else:
            raise BaseException("Model not found!")

        # for i, (v, b, w, e, attr, q, s, a, img_id, ope, ans) in enumerate(train_loader):
        for i, (v, b, w, e, attr, q, s, a, img_id, ans) in enumerate(train_loader):
            v = v.to(device)  # features, Tensor: (64, 46, 2048)
            b = b.to(device)  # spatials, Tensor: (64, 46, 6)
            e = e.to(device)  # entity, Tensor: (64, 7)
            w = w.to(device)  # stat_features, Tensor: (64, 30)
            # attr = attr.to(device)  # attr_features
            q = q.to(device)  # question, Tensor: (64, 12)
            a = a.to(device)  # target, Tensor: (64, 1533)
            # 20250722 add: get ope
            # ope = ope.to(device)  # 20250722: (64, 46, 1024)
            ans = ans.to(device)  # ans, Tensor: (64, 2)
            # sample: list: 9
            # 20250722 add ope
            # sample = [w, q, a, attr, e, ans, v, b, s, ope]  # s: sent
            sample = [w, q, a, attr, e, ans, v, b, s]  # s: sent

            if i < num_batches - 1 and (i + 1) % update_freq > 0:
                trainer.train_step(sample, args, update_params=False)
            else:
                loss, grad_norm, batch_score, batch_question_type_score = trainer.train_step(sample, args, update_params=True)
                total_norm += grad_norm
                count_norm += 1

                total_loss += loss.item()
                train_score += batch_score
                num_updates += 1
                if num_updates % int(args.print_interval / update_freq) == 0:
                    print("Iter: {}, Loss {:.4f}, Norm: {:.4f}, Total norm: {:.4f}, Num updates: {}, Wall time: {:.2f},"
                          "ETA: {}, Train Score {:.4f}".format(i + 1, total_loss / ((num_updates + 1)), grad_norm, total_norm, num_updates,
                                           time.time() - wall_time_start, utils.time_since(t, i / num_batches), train_score / ((num_updates + 1))))
                    if args.testing:
                        break
        total_loss /= num_updates
        train_score = 100 * train_score / (num_updates * args.batch_size)
        train_question_type_score = 100 * train_question_type_score / (num_updates * args.batch_size)
        wandb.log({"train_loss": total_loss, "train_score": train_score})

        if eval_loader is not None:
            print("Evaluating...")
            trainer.model.train(False)
            # eval_cfrf_score: sum of the batch_score calculated by comparing the fusion_pred and ground truth
            # fusion_pred: weighted combination of ban_logits and lxmert_logits
            eval_cfrf_score, fg_score, coarse_score, ens_score, bound, eval_loss = evaluate(model, eval_loader, args, criterion)
            trainer.model.train(True)
            wandb.log({"eval_cfrf_score": eval_cfrf_score, "eval_loss": eval_loss})

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f, question type score: %.2f' %
                     (total_loss, total_norm/count_norm, train_score, train_question_type_score))
        if eval_loader is not None:
            logger.write('\tCFRF score: %.2f (%.2f)' % (100 * eval_cfrf_score, 100 * bound))

        # Save per epoch
        if epoch >= saving_epoch:
            # model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            # utils.save_model(model_path, model, epoch, trainer.optimizer)
            # Save best epoch
            if eval_loader is not None and eval_cfrf_score > best_eval_score:
                model_path = os.path.join(output, 'model_epoch_best.pth')
                utils.save_model(model_path, model, epoch, trainer.optimizer)
                best_eval_score = eval_cfrf_score
        epoch_end_time = time.time()
        epoch_seconds = epoch_end_time - t
        epoch_training_time = time.strftime("%H:%M:%S", time.gmtime(epoch_seconds))
        print(f"Total training time for epoch {epoch+1}/{num_epochs}: {epoch_training_time}")



# core function
def evaluate(model, dataloader, args, criterion):
    device = args.device
    csv_path = args.output
    cfrf_score = 0
    ens_score = 0
    fg_score = 0
    coarse_score = 0
    upper_bound = 0
    num_data = 0
    eval_losses = 0
    n = 0  # num of batches in one epoch
    # df = pd.DataFrame(columns=["Img_id", "Questions", "Answers", "Predictions"])
    batch_dfs = []
    with torch.no_grad():
        # features, spatials, stat_features, entity, attr_features, question, sent (tuple), target, imgid (tuple), ans
        for i, (v, b, w, e, attr, q, s, a, imgid, ans) in enumerate(dataloader):
        # for i, (v, b, w, e, attr, q, s, a, imgid, ope, ans) in enumerate(dataloader):  # s: question sentence
            v = v.to(device)  # image feature
            b = b.to(device)  # spatial feature
            w = w.to(device)  # statistic word feature, tensor
            q = q.to(device)  # objects in each image, tensor
            a = a.to(device)  # target tensor, Tensor: (256, 1533)
            e = e.to(device)  # entities token tensor, i.e., keywords in the image, e.g. ['picture', 'in']
            # ope = ope.to(device)
            ans = ans.to(device)  # answer token tensor, e.g. tensor([1355, 2931], dtype=torch.int32)
            attr = attr.to(device)  # attribute word feature, tensor
            final_preds = None

            # get the labels of answers, to simplify the computation
            a_logits = torch.max(a, 1)[1].data  # Tensor: (256,)
            a_logits = a_logits.cpu().numpy()

            if args.model == 'CFRF_Model':
                # logits, lxmert_logit, ban_logits: weighted combination of lxmert_logit and ban_logits, lxmert_logit and ban_logit
                # sample = [w, q, a, attr, e, ans, v, b, s]  # s: sent
                fusion_preds, lxmert_preds, ban_preds = model(v, b, q, s, e, w, args)  # Tensor: (256, 1533)
                # fusion_preds, lxmert_preds, ban_preds = model(v, b, q, s, e, w, ope, args)
                ens_preds = fusion_preds + lxmert_preds + ban_preds

                # 20250720 add: calculate the loss
                # Fusion loss
                loss_1 = criterion(fusion_preds.float(), a)
                loss_1 /= a.size()[0]

                # Ban loss
                loss_2 = criterion(ban_preds.float(), a)
                loss_2 /= a.size()[0]

                # Lxmert loss
                loss_3 = criterion(lxmert_preds.float(), a)
                loss_3 /= a.size()[0]

                # Total loss
                eval_loss = 0.1 * loss_1 + loss_2 + loss_3  # self.args.fusion_ratio=0.1
                # calculate the matched labels
                fg_score += compute_score_with_logits(ban_preds, a).sum()  # tensor(191., device='cuda:0')
                coarse_score += compute_score_with_logits(lxmert_preds, a).sum()  # tensor(174., device='cuda:0')
                ens_score += compute_score_with_logits(ens_preds, a).sum()  # Tensor: (256, 1533)
                final_preds = fusion_preds
                eval_losses += eval_loss.item()
                n += 1
            else:
                raise BaseException("Model not found!")

            # 20250703: get the batch_preds, to write the answers and predictions into csv file
            if args.write_csv:
                # batch_score: the accuracy of the prediction; batch_preds: the predicted labels
                batch_score, batch_preds = compute_score_with_logits(final_preds, a, write=True)
                batch_score = batch_score.sum()  # tensor(212., device='cuda:0')

                # get the answers in natural language
                # with open('/scratch/xcwx3620/MSc_Project/Codes/CFR_VQA-main-zhd/data/gqa/cache/label2ans.pkl/label2ans.pkl', "rb") as trainval_label2ans_file:
                # with open('D:/CFR_train_extract_0711/cache/label2ans.pkl', "rb") as trainval_label2ans_file:
                with open('./data/cache/label2ans.pkl', "rb") as trainval_label2ans_file:
                    trainval_label2ans = pickle.load(trainval_label2ans_file)
                    # trainval_label2ans_keys = trainval_label2ans.keys()
                    a_strs = [trainval_label2ans[key] for key in
                              a_logits]  # list: 256, e.g. ['cloudless', 'microwave', ...]
                    pred_strs = [trainval_label2ans[key] for key in batch_preds]

                batch_df = pd.DataFrame({
                    "Img_id": list(imgid),
                    "Questions": list(s),
                    "Answers": a_strs,
                    "Predictions": pred_strs
                })

                batch_dfs.append(batch_df)
            batch_scores = compute_score_with_logits(final_preds, a).sum()  # tensor(212., device='cuda:0')
            cfrf_score += batch_scores  # tensor(212., device='cuda:0')
            upper_bound += (a.max(1)[0]).sum()  # tensor(256., device='cuda:0')
            num_data += final_preds.size(0)

    cfrf_score = cfrf_score / len(dataloader.dataset)  # tensor(0.7586, device='cuda:0')
    fg_score = fg_score / len(dataloader.dataset)  # tensor(0.6994, device='cuda:0')
    coarse_score = coarse_score / len(dataloader.dataset)  # tensor(0.6444, device='cuda:0')
    ens_score = ens_score / len(dataloader.dataset)  # tensor(0.7645, device='cuda:0')
    upper_bound = upper_bound / len(dataloader.dataset)
    eval_losses = eval_losses / n

    # write the csv file
    if args.write_csv:
        result_df = pd.concat(batch_dfs, ignore_index=True)
        result_df.to_csv(csv_path, index=False)

    return cfrf_score, fg_score, coarse_score, ens_score, upper_bound, eval_losses

