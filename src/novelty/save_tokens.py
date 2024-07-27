if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")

    args = parser.parse_args()

    assert args.config_path is not None

    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)

    config = get_namespace(config)
    train_df = pd.read_csv(config.general.train_path)
    train_df["indices"] = np.array([i for i in range(len(train_df))])
    train_df = train_df.dropna()
    #train_df = train_df.sample(n=1000)
    train_df.to_csv("sample.csv", index=False)
    print("Started writing tensors")
    save_tokens(config, df=train_df)
    print("Wrote tensors")

    train(config=config, df=train_df)