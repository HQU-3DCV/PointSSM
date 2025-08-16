import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from .defaults import create_ddp_model
import utils.comm as comm
from datasets import build_dataset, collate_fn
from models import build_model
from utils.logger import get_root_logger
from utils.registry import Registry
from utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

VISUALIZER = Registry("visualizer")


class VisualizerBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def get_input_grad(self, feat, pred):
        out_size = pred.size()
        central_point = torch.nn.functional.relu(pred[out_size[0] // 2, :]).sum()
        grad = torch.autograd.grad(central_point, feat)[0]
        grad = torch.nn.functional.relu(grad)
        # grad = grad / (grad.norm() + 1e-6)
        aggregated = grad.sum(1)

        return aggregated

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@VISUALIZER.register_module()
class ERFVisualizer(VisualizerBase):
    def visualize(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Visualization >>>>>>>>>>>>>>>>")

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "visualization")
        make_dirs(save_path)

        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "visual"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "visual"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "visual", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "visual", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "visual", "test", "visual.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        # meter = AverageMeter()
        for idx, data_dict in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            vis_save_path = os.path.join(save_path, "{}_vis".format(data_name))

            grad_map = torch.zeros(segment.size).cuda()
            for i in range(1):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])

                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                input_dict['feat'].requires_grad_()
                idx_part = input_dict["index"]
                pred_part = self.model(input_dict)["seg_logits"]  # (n, k)

                grad_map_part = self.get_input_grad(input_dict['feat'], pred_part)

                np.savez(vis_save_path + '_grad_map',
                         attr=grad_map_part.cpu().numpy(),
                         coord=input_dict['coord'].cpu().numpy())
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                # bs = 0
                # for be in input_dict["offset"]:
                #     grad_map[idx_part[bs:be]] += grad_map_part[bs:be]
                #     bs = be

        logger.info("<<<<<<<<<<<<<<<<< End Visualization <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
