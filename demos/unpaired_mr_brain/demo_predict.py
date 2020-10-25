from deepreg.predict import predict

name = "unpaired_mr_brain"


def main():
    print(
        "The prediction can also be launched using the following command."
        "deepreg_predict --gpu ''"
        f"--config_path demos/{name}/{name}.yaml "
        f"--ckpt_path demos/{name}/dataset/pretrained/learn2reg_t4_unpaired_weakly_train_logs2/save/weights-epoch200.ckpt "
        f"--log_root demos/{name} "
        "--log_dir logs_predict"
        "--save_png --mode test"
    )

    log_root = f"demos/{name}"
    log_dir = "logs_predict"
    ckpt_path = f"{log_root}/dataset/pretrained/learn2reg_t4_unpaired_weakly_train_logs2/save/weights-epoch200.ckpt"
    config_path = f"{log_root}/{name}.yaml"

    predict(
        gpu="0",
        gpu_allow_growth=False,
        ckpt_path=ckpt_path,
        mode="test",
        batch_size=1,
        log_root=log_root,
        log_dir=log_dir,
        sample_label="all",
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
