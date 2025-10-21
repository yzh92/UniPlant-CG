import argparse
import pandas as pd
from compute_result import all_detail_test
from config import cfg

file_root = '/home/yzh_92/Datasets/newClassification/opendomain_main'

def main(task_names,args):
    
    if args.op_flag == "only_opendomain":
        path_root = f'{file_root}/comparision/only_opendomain'
    elif args.op_flag == "all_data":
        path_root = f'{file_root}/comparision/oridata_opendomain'
    else:
        raise ValueError("Invalid op_flag specified.")

    weights_path = f'/home/yzh_92/model/TransReID_res/logs/'

    model_paths = {
        "vit": {
            "model_type": "transformer",
            "plant": "plant/transformer_best.pth",
            "disease": "disease/transformer_best.pth",
            "severity": "severity/transformer_best.pth"
        },
    }

    test_query_path = f"{path_root}/alltestquery"
    plant_path = f"{path_root}/opendomain_plant"
    disease_path = f"{path_root}/opendomain_disease"
    severity_path = f"{path_root}/opendomain_severity"


    for name in task_names:
        if name not in model_paths:
            print(f"[跳过] 未知模型名称：{name}")
            continue

        config = model_paths[name]
        all_detail_test(
            name, config['model_type'],
            plant_path, disease_path, severity_path, test_query_path,
            f"{weights_path}{config['plant']}",
            f"{weights_path}{config['disease']}",
            f"{weights_path}{config['severity']}",
            args,
            True
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation')
    parser.add_argument('--tasks', type=str, nargs='+', required=True,help='List of model names to test, e.g. res18_trans vit')
    parser.add_argument('--op_flag', type=str,default='', help='mode of test opendomian [only_opendomain,all_data,reduce]')
    parser.add_argument('--reduce_rate', type=str, default='', help='reduce rate of test opendomian [30,60,90]')

    parser.add_argument('--dino_flag', type=str,default='', help='mode of test opendomian [only_opendomain,all_data,reduce]')
    parser.add_argument('--domain', type=str,default='', help='mode of test opendomian [only_opendomain,all_data,reduce]')

    parser.add_argument('--output_dir', default='./logs/disease_res', help='Directory to save output files')
    parser.add_argument("--config_file", default="configs/disease_dataset/vit_Uniplant_cg_test.yml", help="", type=str)
    parser.add_argument("--MODEL.DEVICE_ID", default="0", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()

    main(args.tasks,args)


# python test_all_opendomain.py --date 0711_vit_dino_finetune --tasks vit_dino_lora_fintune --op_flag  only_opendomain --output_dir ./visual_dino_fintune/acc_result721
# python test_all_opendomain.py --date 0711_vit_dino_finetune --tasks vit_dino_lora_fintune --op_flag  all_data --output_dir ./visual_dino_fintune/acc_result721