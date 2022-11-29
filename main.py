import sys
import os
import os.path as osp
import copy
import time
import shutil
import cProfile
import logging
from pathlib import Path
import numpy as np
import random
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import os
import inclearn.prune as pruning

os.environ['CUDA_VISIBLE_DEVICES']='0'

repo_name = 'TCIL'
base_dir = osp.realpath(".")[:osp.realpath(".").index(repo_name) + len(repo_name)]
sys.path.insert(0, base_dir)

from sacred import Experiment
ex = Experiment(base_dir=base_dir, save_git_info=False)


import torch

from inclearn.tools import factory, results_utils, utils
from inclearn.learn.pretrain import pretrain
from inclearn.tools.metrics import IncConfusionMeter

def initialization(config, seed, mode, exp_id):

    torch.backends.cudnn.benchmark = True  # This will result in non-deterministic results.
    # ex.captured_out_filter = lambda text: 'Output capturing turned off.'
    cfg = edict(config)
    utils.set_seed(cfg['seed'])
    if exp_id is None:
        exp_id = -1
        cfg.exp.savedir = "./logs"
    logger = utils.make_logger(f"exp{exp_id}_{cfg.exp.name}_{mode}", savedir=cfg.exp.savedir)

    # Tensorboard
    exp_name = f'{exp_id}_{cfg["exp"]["name"]}' if exp_id is not None else f'../inbox/{cfg["exp"]["name"]}'
    tensorboard_dir = cfg["exp"]["tensorboard_dir"] + f"/{exp_name}"

    # If not only save latest tensorboard log.
    # if Path(tensorboard_dir).exists():
    #     shutil.move(tensorboard_dir, cfg["exp"]["tensorboard_dir"] + f"/../inbox/{time.time()}_{exp_name}")

    tensorboard = SummaryWriter(tensorboard_dir)

    return cfg, logger, tensorboard


@ex.command
def train(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
    ex.logger.info(cfg)
    cfg.data_folder = osp.join(base_dir, "data")

    start_time = time.time()
    _train(cfg, _run, ex, tensorboard)
    ex.logger.info("Training finished in {}s.".format(int(time.time() - start_time)))


def _train(cfg, _run, ex, tensorboard):
    device = factory.set_device(cfg)
    trial_i = cfg['trial']

    inc_dataset = factory.get_data(cfg, trial_i)
    ex.logger.info("classes_order")
    ex.logger.info(inc_dataset.class_order)

    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)

    if _run.meta_info["options"]["--file_storage"] is not None:
        _save_dir = osp.join(_run.meta_info["options"]["--file_storage"], str(_run._id))
    else:
        _save_dir = cfg["exp"]["ckptdir"]

    results = results_utils.get_template_results(cfg)

    for task_i in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()

        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
        )

        model.before_task(task_i, inc_dataset)
        # TODO: Move to incmodel.py
        if 'min_class' in task_info:
            ex.logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))

        # Pretraining at step0 if needed
        if task_i == 0 and cfg["start_class"] > 0:
            do_pretrain(cfg, ex, model, device, train_loader, test_loader)
            inc_dataset.shared_data_inc = train_loader.dataset.share_memory
        elif task_i < cfg['start_task']:
            state_dict = torch.load(f'./{cfg.exp.saveckpt}/step{task_i}.ckpt')
            model._parallel_network.load_state_dict(state_dict)
            inc_dataset.shared_data_inc = train_loader.dataset.share_memory
        else:
            model.train_task(train_loader, val_loader)
        model.after_task(task_i, inc_dataset)

        ex.logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))

        ypred, ytrue = model.eval_task(test_loader)

        
        acc_stats = utils.compute_accuracy(ypred, ytrue, increments=model._increments, n_classes=model._n_classes)

        #Logging
        model._tensorboard.add_scalar(f"taskaccu/trial{trial_i}", acc_stats["top1"]["total"], task_i)

        _run.log_scalar(f"trial{trial_i}_taskaccu", acc_stats["top1"]["total"], task_i)
        _run.log_scalar(f"trial{trial_i}_task_top5_accu", acc_stats["top5"]["total"], task_i)

        ex.logger.info(f"top1:{acc_stats['top1']}")
        ex.logger.info(f"top5:{acc_stats['top5']}")

        results["results"].append(acc_stats)

    top1_avg_acc, top5_avg_acc = results_utils.compute_avg_inc_acc(results["results"])

    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"] = top1_avg_acc
    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"] = top5_avg_acc
    ex.logger.info("Average Incremental Accuracy Top 1: {} Top 5: {}.".format(
        _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"],
        _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"],
    ))
    if cfg["exp"]["name"]:
        results_utils.save_results(results, cfg["exp"]["name"])


def do_pretrain(cfg, ex, model, device, train_loader, test_loader):
    if not os.path.exists(osp.join(ex.base_dir, 'pretrain/')):
        os.makedirs(osp.join(ex.base_dir, 'pretrain/'))
    model_path = osp.join(
        ex.base_dir,
        "pretrain/{}_{}_cosine_{}_dynamic_{}_nplus1_{}_{}_trial_{}_{}_seed_{}_start_{}_epoch_{}.pth".format(
            cfg["model"],
            cfg["convnet"],
            cfg["weight_normalization"],
            cfg["dea"],
            cfg["div_type"],
            cfg["dataset"],
            cfg["trial"],
            cfg["train_head"],
            cfg['seed'],
            cfg["start_class"],
            cfg["pretrain"]["epochs"],
        ),
    )
    if osp.exists(model_path):
        print("Load pretrain model")
        if hasattr(model._network, "module"):
            model._network.module.load_state_dict(torch.load(model_path))
        else:
            model._network.load_state_dict(torch.load(model_path))
    else:
        pretrain(cfg, ex, model, device, train_loader, test_loader, model_path)

@ex.command
def test(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "test", _run._id)
    ex.logger.info(cfg)

    trial_i = cfg['trial']
    cfg.data_folder = osp.join(base_dir, "data")
    inc_dataset = factory.get_data(cfg, trial_i)

    ex.logger.info("classes_order")
    ex.logger.info(inc_dataset.class_order)

    # inc_dataset._current_task = taski
    # train_loader = inc_dataset._get_loader(inc_dataset.data_cur, inc_dataset.targets_cur)
    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    model._network.task_size = cfg.increment

    test_results = results_utils.get_template_results(cfg)

    for taski in range(inc_dataset.n_tasks):
        task_info, train_loader, _, test_loader = inc_dataset.new_task()
        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=task_info["max_task"]
        )
        model.before_task(taski, inc_dataset)
        state_dict = torch.load(f'./{cfg.exp.saveckpt}/step{taski}.ckpt')
        if cfg.get("caculate_params", False):
            model._parallel_network.load_state_dict(state_dict,False)
        else:
            model._parallel_network.load_state_dict(state_dict)
        
        model.eval()

        #Build exemplars
        model.after_task(taski, inc_dataset)

        
        ypred, ytrue = model.eval_task(test_loader)

        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=model._increments, n_classes=model._n_classes)
        
        test_acc_task_stats = utils.compute_old_new_mix(ypred, ytrue, increments=model._increments, n_classes=model._n_classes, task_order=inc_dataset.class_order)
        
        test_results['results'].append(test_acc_stats)
        ex.logger.info(f"task{taski} test acc:{test_acc_stats['top1']}")

        # ex.logger.info(f"task{taski} task mean:{test_acc_task_stats['task_mean']} \ntest class\n:{test_acc_task_stats['class_info']} \ntest task\n:{test_acc_task_stats['task_info']}")
        ex.logger.info(f"task{taski} all task mean:{test_acc_task_stats['task_mean']} \n task means: {test_acc_task_stats['task_means']} \n new err:{test_acc_task_stats['new_err']} \nold err:{test_acc_task_stats['old_err']}\nnew_old_err:{test_acc_task_stats['new_old_err']} \nold_new_err:{test_acc_task_stats['old_new_err']} \nerr_among_task:{test_acc_task_stats['err_among_task']} \nerr_inner_task:{test_acc_task_stats['err_inner_task']}")

    avg_test_acc = results_utils.compute_avg_inc_acc(test_results['results'])
    ex.logger.info(f"Test Average Incremental Accuracy: {avg_test_acc}")


@ex.command
def prune(_run, _rnd, _seed):
    from copy import deepcopy
        
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "prune", _run._id)
    #ex.logger.info(cfg)

    trial_i = cfg['trial']
    cfg.data_folder = osp.join(base_dir, "data")
    inc_dataset = factory.get_data(cfg, trial_i)

    #ex.logger.info("classes_order")
    #ex.logger.info(inc_dataset.class_order)

    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    tmodel = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)

    model._network.task_size = cfg.increment
    tmodel._network.task_size = cfg.increment

    test_results = results_utils.get_template_results(cfg)


    for taski in range(inc_dataset.n_tasks):

        print(f"--------------对step{taski}进行剪枝--------------")

        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()


        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
        )

        model.before_task(taski, inc_dataset)

        tmodel.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
        )

        tmodel.before_task(taski, inc_dataset)

        state_dict = torch.load(f'./{cfg.exp.saveckpt}/step{taski}.ckpt')
        tmodel._parallel_network.load_state_dict(state_dict)
        model._parallel_network.module.convnets[-1].load_state_dict(tmodel._parallel_network.module.convnets[-1].state_dict())
        
        net = model._parallel_network.module.convnets[-1]

        flops_raw, params_raw = pruning.get_model_complexity_info(
            net, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
        print(f'-pruning step{taski} with net{taski}')
        print('-[INFO] before pruning flops:  ' + flops_raw)
        print('-[INFO] before pruning params:  ' + params_raw)
        # 选择裁剪方式
        mod = 'fpgm'

        # 剪枝引擎建立
        slim = pruning.Autoslim(net, inputs=torch.randn(
            1, 3, 32, 32), compression_ratio=0.4)

        if mod == 'fpgm':
            config = {
                'layer_compression_ratio': None,
                'norm_rate': 1.0, 'prune_shortcut': 1,
                'dist_type': 'l1', 'pruning_func': 'fpgm'
            }
        elif mod == 'l1':
            config = {
                'layer_compression_ratio': None,
                'norm_rate': 1.0, 'prune_shortcut': 1,
                'global_pruning': False, 'pruning_func': 'l1'
            }
        slim.base_prunging(config)
        flops_new, params_new = pruning.get_model_complexity_info(
            net, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
        print('-[INFO] after pruning flops:  ' + flops_new)
        print('-[INFO] after pruning params:  ' + params_new)
    
        model.after_prune(taski, inc_dataset)

        model.set_optimizer()

        model.train_task(train_loader, val_loader)

        model.eval()

        model.after_task(taski, inc_dataset)
        
        model._parallel_network = model._parallel_network.cuda()
        ypred, ytrue = model.eval_task(test_loader)

        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=model._increments, n_classes=model._n_classes)
        
        test_results['results'].append(test_acc_stats)
        ex.logger.info(f"task{taski} test acc:{test_acc_stats['top1']}")
        ex.logger.info(f"top1:{test_acc_stats['top1']}")
        ex.logger.info(f"top5:{test_acc_stats['top5']}")

        save_path = os.path.join(os.getcwd(), f"{cfg.exp.saveckpt}")
        torch.save(model._parallel_network.cpu(), "{}/prune_step{}.ckpt".format(save_path, taski)) # 保存整个神经网络的模型结构以及参数


    top1_avg_acc, top5_avg_acc = results_utils.compute_avg_inc_acc(test_results["results"])

    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top1"] = top1_avg_acc
    _run.info[f"trial{trial_i}"][f"avg_incremental_accu_top5"] = top5_avg_acc
    ex.logger.info("Average Incremental Accuracy Top 1: {} Top 5: {}.".format(
        top1_avg_acc,
        top5_avg_acc,
    ))


if __name__ == "__main__":
    ex.add_config("./configs/cifar_b0_10s.yaml")
    ex.run_commandline()
