import os.path
import os.path as osp
from .evaluator import Eval_thread
from .dataloader import EvalDataset, EvalVideoDataset
import tqdm
import torch

def evaluate(args):

    torch.set_num_threads(3)

    pred_dir = args.save_test_path_root     # preds/
    test_name = args.test_name
    output_dir = args.save_dir              # ./
    gt_dir = args.data_root                 # ./data/
    dataset = args.trainset
    method = args.methods

    threads = []
    test_paths = args.test_paths.split('+') # Cube360
    for dataset_setname in tqdm.tqdm(test_paths):

        dataset_name = dataset_setname.split('/')[0]

        pred_dir_all = osp.join(pred_dir, dataset_name)     # preds/Cube360
        gt_dir_all = os.path.join(gt_dir, dataset, "test", "labels")

        if dataset_name == "PAVS":
            loader = EvalVideoDataset(pred_dir_all, gt_dir_all, test_name)
        else:
            loader = EvalDataset(pred_dir_all, gt_dir_all)
        
        thread = Eval_thread(loader, method, dataset_setname, output_dir, cuda=True)
        threads.append(thread)

    for thread in threads:
        print(thread.run())
