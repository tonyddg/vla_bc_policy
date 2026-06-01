import torch

from vla_bc_policy.dataset import Pi0LeRobotDataModule, FetchStandardCameraInfos, camera_info_list_to_dict_list

from vla_bc_policy.model.lit_model import PolicyModule
from vla_bc_policy.train import get_trainer

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    dm = Pi0LeRobotDataModule(
        "trajectories_tidy_house_all_bc_fix_la_state_rot_6d_action_axis_angle",
        camera_info_list_to_dict_list(FetchStandardCameraInfos),
        debug_config = False,

        batch_size = 1024,
        num_workers = 32,
    )
    pm = PolicyModule.load_from_checkpoint("output/res_mlp/version_1/checkpoints/last.ckpt")
    
    trainer = get_trainer(
        "res_mlp", precision = "32", 
        max_epochs = 200, patience = None,
        gradient_clip_val = 2.0
    )
    trainer.fit(
        pm, 
        datamodule = dm, 
        # ckpt_path = "output/res_mlp/version_1/checkpoints/last.ckpt"
    )
