import os
from config import cfg
import argparse
from datasets import make_dataloader,make_test_dataloader
from model import make_model
from processor import do_inference,do_altogether_inference,do_altogether_inference_detail,do_altogether_inference_all_imgpath
from utils.logger import setup_logger
import warnings
import numpy as np
import torch
import json
warnings.filterwarnings('ignore',category=UserWarning)
from collections import defaultdict
import math

def all_detail_test(model_flag_name,model_type,plant_path,disease_path,severity_path,test_query_path,plant_weights,disease_weights,severity_weights,args,args_flag=False,
domain_flag = 'indomain'):
    if not args_flag:
        parser = argparse.ArgumentParser(description="Uniplant-CG Baseline Training")

        parser.add_argument('--reduce_rate', type=str, required=True,
                            help='reduce rate: e.g. reduce_30percent/')
        parser.add_argument(
                "--config_file", default="configs/disease_dataset/vit_Uniplant_cg.yml", help="", type=str
            )
        parser.add_argument(
                "--MODEL.DEVICE_ID", default="0", help="path to config file", type=str
            )
        parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                                nargs=argparse.REMAINDER)

        args = parser.parse_args()


    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    

    output_dir = args.output_dir

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("disease_classfication", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    all_query_root = test_query_path
    plant_model_dataroot = plant_path
    plant_disease_model_dataroot = disease_path
    plant_severity_model_dataroot = severity_path


    _, _, plant_val_loader, plant_num_query, plant_num_classes, plant_camera_num, plant_view_num = make_test_dataloader(cfg,plant_model_dataroot,all_query_root)
    _, _, plant_disease_val_loader, plant_disease_num_query, plant_disease_num_classes, plant_disease_camera_num,plant_disease_view_num = make_test_dataloader(cfg,plant_disease_model_dataroot,all_query_root)
    _, _, plant_severity_val_loader, plant_severity_num_query, plant_severity_num_classes, plant_severity_camera_num,plant_severity_view_num = make_test_dataloader(cfg,plant_severity_model_dataroot,all_query_root)

    print(f"numbers of plant:{plant_num_classes}")
    print(f"numbers of disease:{plant_disease_num_classes}")
    print(f"numbers of severity:{plant_severity_num_classes}")
    
    plant_model_weights_path = plant_weights
    cfg.MODEL.NAME = model_type
    plant_model = make_model(cfg, num_class=plant_num_classes, camera_num=plant_camera_num, view_num = plant_view_num)
    

    if args.dino_flag == 'dino':
        plant_model.load_param(plant_model_weights_path,dino_finetune_flag=True)
    elif args.dino_flag == 'vit':
        plant_model.load_param(plant_model_weights_path)
    else:
        plant_model.load_param(plant_model_weights_path)

    plant_rankpids,plant_pids,plant_rank_imgpaths,plant_query_imgpaths = do_altogether_inference_all_imgpath(cfg,
                 plant_model,
                 plant_val_loader,
                 plant_num_query)

    print("plant model=========================")
    print("qeury lenth:",len(plant_pids))
    print(f"[DEBUG] plant_query_imgpaths : {plant_query_imgpaths[:10]}")
    print(f"[DEBUG] plant_pids : {plant_pids[:10]}")

    plant_disease_model_weights_path = disease_weights
    cfg.MODEL.NAME = model_type
    plant_disease_model = make_model(cfg, num_class=plant_disease_num_classes, camera_num=plant_disease_camera_num, view_num = plant_disease_view_num)

    if args.dino_flag == 'dino':
        plant_disease_model.load_param(plant_disease_model_weights_path,dino_finetune_flag=True)
    elif args.dino_flag == 'vit':
        plant_disease_model.load_param(plant_disease_model_weights_path)
    else:
        plant_disease_model.load_param(plant_disease_model_weights_path)
    
    plant_disease_rankpids,plant_disease_pids,disease_rank_imgpaths,disease_query_imgpaths= do_altogether_inference_all_imgpath(cfg,
                 plant_disease_model,
                 plant_disease_val_loader,
                 plant_disease_num_query)

    print("disease qeury lenth:",len(plant_disease_pids))
    print(f"[DEBUG] disease_rank_imgpaths : {disease_rank_imgpaths[:10]}")
    print(f"[DEBUG] disease_query_imgpaths : {disease_query_imgpaths[:10]}")

    plant_severity_model_weights_path = severity_weights
    cfg.MODEL.NAME = model_type
    plant_severity_model = make_model(cfg, num_class=plant_severity_num_classes, camera_num=plant_severity_camera_num, view_num = plant_severity_view_num)

    if args.dino_flag == 'dino':
        plant_severity_model.load_param(plant_severity_model_weights_path,dino_finetune_flag=True)
    elif args.dino_flag == 'vit':
        plant_severity_model.load_param(plant_severity_model_weights_path)
    else:
        plant_severity_model.load_param(plant_severity_model_weights_path)

    plant_severity_rankpids,plant_severity_pids,plant_severity_imgpaths= do_altogether_inference_detail(cfg,
                 plant_severity_model,
                 plant_severity_val_loader,
                 plant_severity_num_query)


    print("severity qeury lenth:",len(plant_severity_pids))
    
    
    result = []

    for single_rank in plant_severity_rankpids:
        result.append(single_rank[0])


    final_count = 0
    temp_disease_count = 0

    plant_rankpids = np.array(plant_rankpids)
    plant_disease_rankpids = np.array(plant_disease_rankpids)
    plant_severity_rankpids = np.array(plant_severity_rankpids)


    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    base_path = '/home/yzh_92/Datasets/newClassification/opendomain_main/classification/base_file'
    disease_indomain = set(load_json(f'{base_path}/disease/disease_indomain.json'))
    disease_outdomain = set(load_json(f'{base_path}/disease/disease_outdomain.json'))

    final_count = 0
    indomain_count = 0
    indomain_total = 0
    outdomain_count = 0
    outdomain_total = 0

    tpr_correct_count = 0  
    tpr_total_count = 0    

    plant_match_list = []
    disease_match_list = []

    for idx, qid in enumerate(plant_pids):
        match_plantid = plant_rankpids[idx][0]
        match_plant_disease_id = ''

        plant_match_list.append({
            "query": plant_query_imgpaths[idx],
            "match": plant_rank_imgpaths[idx][0]
        })

        for rank_idx, plant_diseaseid in enumerate(plant_disease_rankpids[idx]):
            imgname = os.path.basename(disease_rank_imgpaths[idx][rank_idx])
            parts = imgname.split('_')  

            id_part = parts[0] 
            split_flag = int(parts[2])  

            plant_id_extracted = id_part[:split_flag]
            disease_id_extracted = id_part[split_flag:]

            if int(plant_id_extracted) == match_plantid:
                match_plant_disease_id = plant_diseaseid

                disease_match_list.append({
                    "query": disease_query_imgpaths[idx],
                    "match": disease_rank_imgpaths[idx][rank_idx]
                })
                break

        match_severity_id = result[idx]
        all_match_id = str(match_plant_disease_id) + str(match_severity_id)

        tpr_total_count += 1
        if str(match_plant_disease_id) in disease_outdomain:
            tpr_correct_count += 1

        if qid == int(all_match_id):
            final_count += 1

            if str(match_plant_disease_id) in disease_indomain:
                indomain_count += 1
            elif str(match_plant_disease_id) in disease_outdomain:
                outdomain_count += 1

        if str(match_plant_disease_id) in disease_indomain:
            indomain_total += 1
        elif str(match_plant_disease_id) in disease_outdomain:
            outdomain_total += 1
        else:
            print(f"Warning: {match_plant_disease_id} not found in either indomain or outdomain disease lists.")

    final_acc = final_count / float(len(plant_pids))
    indomain_acc = indomain_count / float(indomain_total) if indomain_total > 0 else 0.0
    outdomain_acc = outdomain_count / float(outdomain_total) if outdomain_total > 0 else 0.0
    
    tpr_accuracy = tpr_correct_count / float(tpr_total_count) if tpr_total_count > 0 else 0.0

    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"In-domain Accuracy: {indomain_acc:.4f} ({indomain_count}/{indomain_total})")
    print(f"Out-domain Accuracy: {outdomain_acc:.4f} ({outdomain_count}/{outdomain_total})")
    print(f"TPR Accuracy (Out-domain Detection): {tpr_accuracy:.4f} ({tpr_correct_count}/{tpr_total_count})")  

    with open(os.path.join(output_dir, f"plant_match_paths_{args.tasks}.json"), "w") as f:
        json.dump(plant_match_list, f, indent=2)

    with open(os.path.join(output_dir, f"disease_match_paths_{args.tasks}.json"), "w") as f:
        json.dump(disease_match_list, f, indent=2)


    plant_class_stat = defaultdict(lambda: [0, 0])      
    disease_class_stat = defaultdict(lambda: [0, 0])    
    severity_class_stat = defaultdict(lambda: [0, 0])   

    severity_prediction_records = []

    for idx,qid in enumerate(plant_pids):
        imagename = plant_query_imgpaths[idx]  
        match_plantid = plant_rankpids[idx][0]

        qid = str(qid)

        split_point = imagename.split('_')[2]
        gt_plant_id = imagename.split('_')[0][:int(split_point)]
        
        gt_disease_id = qid[:-1]

        gt_severity = qid[-1]

        pred_plant_id = str(plant_rankpids[idx][0])
        pred_disease_id = None

        for rank_idx, plant_diseaseid in enumerate(plant_disease_rankpids[idx]):
            imgname = os.path.basename(disease_rank_imgpaths[idx][rank_idx])
            
            parts = imgname.split('_')  
            
            id_part = parts[0]  
            split_flag = int(parts[2])  

            plant_id_extracted = id_part[:split_flag]     
            disease_id_extracted = id_part[split_flag:]   

            if int(plant_id_extracted) == match_plantid:
                pred_disease_id = plant_diseaseid
                break

        pred_severity = str(result[idx])
        image_path = plant_severity_imgpaths[idx]  

        severity_prediction_records.append({
            "image_path": image_path,
            "gt_severity": gt_severity,
            "pred_severity": pred_severity
        })

        plant_class_stat[gt_plant_id][1] += 1
        if pred_plant_id == gt_plant_id:
            plant_class_stat[gt_plant_id][0] += 1

        disease_class_stat[gt_disease_id][1] += 1
        if pred_disease_id == int(gt_disease_id):
            disease_class_stat[gt_disease_id][0] += 1

        severity_class_stat[gt_severity][1] += 1
        if pred_severity == gt_severity:
            severity_class_stat[gt_severity][0] += 1

    def compute_accuracy(stat_dict):
        return {k: round(v[0] / v[1], 4) if v[1] > 0 else 0.0 for k, v in stat_dict.items()}

    plant_acc_dict = compute_accuracy(plant_class_stat)
    disease_acc_dict = compute_accuracy(disease_class_stat)
    severity_acc_dict = compute_accuracy(severity_class_stat)

    with open(os.path.join(output_dir, "per_plant_accuracy.json"), "w") as f:
        json.dump(plant_acc_dict, f, indent=4)

    with open(os.path.join(output_dir, "per_disease_accuracy.json"), "w") as f:
        json.dump(disease_acc_dict, f, indent=4)

    with open(os.path.join(output_dir, "per_severity_accuracy.json"), "w") as f:
        json.dump(severity_acc_dict, f, indent=4)

    with open(os.path.join(output_dir, "severity_predictions.json"), "w") as f:
        json.dump(severity_prediction_records, f, indent=4)


    compute_indomain_outdomain_accuracy(
        plant_acc_dict=plant_acc_dict,
        disease_acc_dict=disease_acc_dict,
        severity_prediction_records=severity_prediction_records,
        plant_disease_imgpaths=disease_query_imgpaths,
        all_acc=final_acc,
        indomain_acc=indomain_acc,
        outdomain_acc=outdomain_acc,
        output_dir=output_dir,
        domain_flag=domain_flag
    )

    print(f"\n Accuracy stats saved to {output_dir}")



def compute_indomain_outdomain_accuracy(
    plant_acc_dict,
    disease_acc_dict,
    severity_prediction_records,
    plant_disease_imgpaths,
    all_acc,
    indomain_acc,
    outdomain_acc,
    output_dir=None,
    domain_flag=None
):

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    base_path = '/home/yzh_92/Datasets/newClassification/opendomain_main/classification/base_file'

    plant_count = load_json(f'{base_path}/plant/plant_category_counts.json')
    plant_indomain = set(load_json(f'{base_path}/plant/plant_indomain.json'))
    plant_outdomain = set(load_json(f'{base_path}/plant/plant_outdomain.json'))

    disease_count = load_json(f'{base_path}/disease/disease_category_counts.json')
    disease_indomain = set(load_json(f'{base_path}/disease/disease_indomain.json'))
    disease_outdomain = set(load_json(f'{base_path}/disease/disease_outdomain.json'))

    def weighted_avg(acc_dict, count_dict, selected_ids):
        total_correct = 0
        total_samples = 0
        for cid in selected_ids:
            if cid in acc_dict and cid in count_dict:
                acc = acc_dict[cid]
                n = count_dict[cid]
                total_correct += acc * n
                total_samples += n
        return round(total_correct / total_samples, 4) if total_samples > 0 else 0.0

    plant_indomain_acc = weighted_avg(plant_acc_dict, plant_count, plant_indomain)
    plant_outdomain_acc = weighted_avg(plant_acc_dict, plant_count, plant_outdomain)
    plant_total_acc = weighted_avg(plant_acc_dict, plant_count, plant_acc_dict.keys())

    disease_indomain_acc = weighted_avg(disease_acc_dict, disease_count, disease_indomain)
    disease_outdomain_acc = weighted_avg(disease_acc_dict, disease_count, disease_outdomain)
    disease_total_acc = weighted_avg(disease_acc_dict, disease_count, disease_acc_dict.keys())

    severity_domain_stat = {
        "indomain": [0, 0],   
        "outdomain": [0, 0],
    }

    print(f"[DEBUG] severity_prediction_records : {severity_prediction_records[:10]}")

    for record in severity_prediction_records:
        severity_image_name = os.path.basename(record["image_path"])
        parts = severity_image_name.split("_")
        severity_suffix = "_".join(parts[1:])  

        matched_disease_img = None
        for disease_img in plant_disease_imgpaths:
            if disease_img.endswith(severity_suffix):
                matched_disease_img = disease_img
                break

        if matched_disease_img is None:
            print(f"Warning: No matching disease image found for {severity_image_name}")
            continue  

        disease_id = matched_disease_img.split("_")[0][:-1]  

        if disease_id in disease_indomain:
            domain_type = "indomain"
        elif disease_id in disease_outdomain:
            domain_type = "outdomain"
        else:
            
            continue  

        severity_domain_stat[domain_type][1] += 1
        if record["gt_severity"] == record["pred_severity"]:
            severity_domain_stat[domain_type][0] += 1

    severity_indomain_acc = round(
        severity_domain_stat["indomain"][0] / severity_domain_stat["indomain"][1], 4
    ) if severity_domain_stat["indomain"][1] > 0 else 0.0

    severity_outdomain_acc = round(
        severity_domain_stat["outdomain"][0] / severity_domain_stat["outdomain"][1], 4
    ) if severity_domain_stat["outdomain"][1] > 0 else 0.0

    severity_total_acc = round(
        sum(x[0] for x in severity_domain_stat.values()) /
        sum(x[1] for x in severity_domain_stat.values()), 4
    ) if sum(x[1] for x in severity_domain_stat.values()) > 0 else 0.0

    if domain_flag=='indomain':
        final_result = {
            "plant": plant_indomain_acc
            "disease":disease_indomain_acc
            "severity": severity_indomain_acc
            "all":indomain_acc
        }
    else:
        final_result = {
            "plant": plant_total_acc
            "disease":disease_total_acc
            "all":outdomain_acc
        }

    if output_dir:
        with open(os.path.join(output_dir, "indomain_outdomain_accuracy.json"), "w") as f:
            json.dump(final_result, f, indent=4)

    print("\n Indomain/Outdomain Accuracy:")
    print(json.dumps(final_result, indent=4))
    return final_result
