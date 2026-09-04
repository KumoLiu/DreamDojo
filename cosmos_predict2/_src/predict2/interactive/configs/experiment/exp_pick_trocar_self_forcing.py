from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.distill.utils.config_helper import (
    build_no_s3_run,
    deep_update_config_dict,
)
from cosmos_predict2._src.predict2.interactive.configs.experiment.exp_action_self_forcing import (
    make_experiment as make_self_forcing_experiment,
)
from cosmos_predict2._src.predict2.interactive.configs.experiment.exp_action_warmup import (
    make_experiment as make_warmup_experiment,
)


TEACHER_CHECKPOINT = (
    "/localhome/local-yunl/DreamDojo/outputs/train/dreamdojo/"
    "pick_trocar_headcam_rollout_holdout_5s5f_ema_both/"
    "g1_pick_trocar_headcam_2b_rollout_holdout_5s5f_ema_both/"
    "checkpoints/iter_000003000"
)
WARMUP_CHECKPOINT = (
    "/localhome/local-yunl/DreamDojo/outputs/train/dreamdojo/"
    "pick_trocar_self_forcing/g1_pick_trocar_iter2500_warmup/"
    "checkpoints/iter_000020000"
)
STRATIFIED_WARMUP_CHECKPOINT = (
    "/localhome/local-yunl/DreamDojo/outputs/train/dreamdojo/"
    "pick_trocar_self_forcing_97f/g1_pick_trocar_iter2500_warmup_stratified/"
    "checkpoints/iter_000003000"
)


warmup_base = make_warmup_experiment(
    name="g1_pick_trocar_iter2500_warmup",
    data="pick_trocar_iter2500_warmup",
    overrides={
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing",
            "name": "g1_pick_trocar_iter2500_warmup",
        },
        "checkpoint": {
            "load_path": TEACHER_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 1000,
        },
    },
)
warmup = build_no_s3_run(OmegaConf.to_container(warmup_base, resolve=False))
deep_update_config_dict(
    warmup,
    {
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing",
            "name": "g1_pick_trocar_iter2500_warmup",
        },
        "checkpoint": {
            "load_path": TEACHER_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 1000,
        },
    },
)
warmup = LazyDict(warmup, flags={"allow_objects": True})


self_forcing_base = make_self_forcing_experiment(
    name="g1_pick_trocar_iter2500_self_forcing",
    data="pick_trocar_rollout_train_long",
    overrides={
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing",
            "name": "g1_pick_trocar_iter2500_self_forcing",
        },
        "checkpoint": {
            "load_path": WARMUP_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 500,
        },
        "model": {
            "config": {
                "teacher_load_from": {
                    "load_path": f"{TEACHER_CHECKPOINT}/model",
                    "credentials": "",
                },
            },
        },
    },
)
self_forcing = build_no_s3_run(OmegaConf.to_container(self_forcing_base, resolve=False))
deep_update_config_dict(
    self_forcing,
    {
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing",
            "name": "g1_pick_trocar_iter2500_self_forcing",
        },
        "checkpoint": {
            "load_path": WARMUP_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 500,
        },
        "model": {
            "config": {
                "teacher_load_from": {
                    "load_path": f"{TEACHER_CHECKPOINT}/model",
                    "credentials": "",
                },
            },
        },
    },
)
self_forcing = LazyDict(self_forcing, flags={"allow_objects": True})

warmup_stratified_base = make_warmup_experiment(
    name="g1_pick_trocar_iter2500_warmup_stratified",
    data="pick_trocar_iter2500_warmup_stratified",
    overrides={
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing_97f",
            "name": "g1_pick_trocar_iter2500_warmup_stratified",
        },
        "checkpoint": {
            "load_path": TEACHER_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 1000,
        },
        "trainer": {"max_iter": 5000},
    },
)
warmup_stratified = build_no_s3_run(
    OmegaConf.to_container(warmup_stratified_base, resolve=False)
)
deep_update_config_dict(
    warmup_stratified,
    {
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing_97f",
            "name": "g1_pick_trocar_iter2500_warmup_stratified",
        },
        "checkpoint": {
            "load_path": TEACHER_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 1000,
        },
        "trainer": {"max_iter": 5000},
    },
)
warmup_stratified = LazyDict(warmup_stratified, flags={"allow_objects": True})

self_forcing_97f_base = make_self_forcing_experiment(
    name="g1_pick_trocar_iter2500_self_forcing_97f",
    data="pick_trocar_rollout_train_97f",
    overrides={
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing_97f",
            "name": "g1_pick_trocar_iter2500_self_forcing_97f",
        },
        "checkpoint": {
            "load_path": STRATIFIED_WARMUP_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 500,
        },
        "model": {
            "config": {
                "teacher_load_from": {
                    "load_path": f"{TEACHER_CHECKPOINT}/model",
                    "credentials": "",
                },
            },
        },
    },
)
self_forcing_97f = build_no_s3_run(
    OmegaConf.to_container(self_forcing_97f_base, resolve=False)
)
deep_update_config_dict(
    self_forcing_97f,
    {
        "job": {
            "project": "dreamdojo",
            "group": "pick_trocar_self_forcing_97f",
            "name": "g1_pick_trocar_iter2500_self_forcing_97f",
        },
        "checkpoint": {
            "load_path": STRATIFIED_WARMUP_CHECKPOINT,
            "load_training_state": False,
            "strict_resume": False,
            "save_iter": 500,
        },
        "model": {
            "config": {
                "teacher_load_from": {
                    "load_path": f"{TEACHER_CHECKPOINT}/model",
                    "credentials": "",
                },
            },
        },
    },
)
self_forcing_97f = LazyDict(self_forcing_97f, flags={"allow_objects": True})


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="pick_trocar_iter2500_warmup_no_s3",
    node=warmup,
)
cs.store(
    group="experiment",
    package="_global_",
    name="pick_trocar_iter2500_self_forcing_no_s3",
    node=self_forcing,
)
cs.store(
    group="experiment",
    package="_global_",
    name="pick_trocar_iter2500_warmup_stratified_no_s3",
    node=warmup_stratified,
)
cs.store(
    group="experiment",
    package="_global_",
    name="pick_trocar_iter2500_self_forcing_97f_no_s3",
    node=self_forcing_97f,
)
