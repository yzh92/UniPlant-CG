import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.metrics_test import test_R1_mAP_eval
from utils.metrics_test_detail import test_R1_mAP_eval as eval_func
from torch.cuda import amp
import torch.distributed as dist
import gc
import subprocess

def gpu_total_used(device=0):
    cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {device}"
    return int(subprocess.check_output(cmd, shell=True).decode().strip())

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    model_category = cfg.MODEL.CATEGORY

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("Uniplant-CG")
    # ===== 临时加 handler =====    
    # if not logger.handlers:          
    #     ch = logging.StreamHandler()
    #     ch.setLevel(logging.INFO)
    #     ch.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
    #     logger.addHandler(ch)
    # logger.setLevel(logging.INFO)    
    logger.info('start training')

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    best_rank1_acc = 0
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        logger.info(f"Epoch {epoch} start - Total iterations this epoch: {len(train_loader)}")
        
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        
        if epoch % eval_period == 0:
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            cached_before    = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[Before Eval]  Allocated: {allocated_before:.2f} GB, Cached: {cached_before:.2f} GB")

            if cfg.MODEL.DIST_TRAIN:
                print('dist training')
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img)
                            evaluator.update((feat.detach().cpu(), vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    pass
            else:
                model.eval()
                with torch.no_grad():
                    logger.info(f"val_loader: {len(val_loader)} batches, batch_size={val_loader.batch_size}, total_samples={len(val_loader.dataset)}")
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img)
                        evaluator.update((feat.detach().cpu(), vid, camid))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if cmc[0]>best_rank1_acc:
                    best_rank1_acc = cmc[0]
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                
                print(f"model:{model_category},Best acc:{best_rank1_acc}")
                gc.collect()    
                torch.cuda.empty_cache()
                
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                cached_after    = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"[After  Eval]  Allocated: {allocated_after:.2f} GB, Cached: {cached_after:.2f} GB")


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _,_,_ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_altogether_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = test_R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP,rank_pids,q_pids ,_, _, _, _, _ = evaluator.compute()

    return rank_pids,q_pids


    
def do_altogether_inference_detail(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = test_R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP,rank_pids,q_pids ,_, _, _, _, _ = evaluator.compute()

    return rank_pids,q_pids,img_path_list

def do_altogether_inference_all_imgpath(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = eval_func(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()


    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid,imgpath))

    cmc, mAP,rank_pids,q_pids ,_, _, _, _, _,rank_imgpaths,query_imgpaths  = evaluator.compute()


    return rank_pids,q_pids,rank_imgpaths,query_imgpaths

import numpy as np

def do_altogether_inference_feat(cfg,
                                 model,
                                 val_loader,
                                 num_query):
    device = "cuda"
    model.eval()
    model.to(device)

    all_feats = []
    all_pids  = []  
    img_path_list = []

    for img, pid, camid, camids, target_view, imgpath in val_loader:
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)          
            all_feats.append(feat.cpu())
            pid_np = np.array(pid)          
            pid_no_last = pid_np
            all_pids.extend(pid_no_last)
            img_path_list.extend(imgpath)

    all_feats = torch.cat(all_feats, dim=0).numpy() 
    all_pids  = np.array(all_pids)                   

    return all_feats, all_pids, img_path_list