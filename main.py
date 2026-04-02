import argparse
import time
import torch
import tiktoken

from model import GPTModel, CommitBugClassifier
from gpt_weights import download_and_load_gpt2, load_weights_into_gpt
from dataset import (prepare_data, build_dataloaders,
                     NUM_INTENT_CLASSES, NUM_SEVERITY_CLASSES)
from train import (train_classifier, calc_accuracy_loader,
                   calc_loss_loader, evaluate_model)
from inference import classify_commit


CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {
    "vocab_size":     50257,
    "context_length": 1024,
    "drop_rate":      0.0,
    "qkv_bias":       True,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.intent_head.parameters():
        param.requires_grad = True
    for param in model.severity_head.parameters():
        param.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Commit Bug Classifier  |  device: {device}")
    print(f"{'='*60}\n")

    tokenizer = tiktoken.get_encoding("gpt2")

    print("[ 1/5 ] Preparing dataset ...")
    prepare_data(raw_data_dir=args.data, output_dir="data")

    train_loader, val_loader, test_loader, max_seq_len = build_dataloaders(
        data_dir="data", tokenizer=tokenizer, batch_size=args.batch_size
    )
    print(f"       max sequence length: {max_seq_len}")
    print(f"       batches -> train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}\n")

    assert max_seq_len <= BASE_CONFIG["context_length"], (
        f"Dataset length {max_seq_len} exceeds model context length {BASE_CONFIG['context_length']}."
    )

    print("[ 2/5 ] Loading GPT-2 weights ...")
    cfg = {**BASE_CONFIG, **MODEL_CONFIGS[CHOOSE_MODEL]}
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    if not args.skip_download:
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        gpt = GPTModel(cfg)
        load_weights_into_gpt(gpt, params)
    else:
        gpt = GPTModel(cfg)
        gpt.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    print("       GPT-2 weights loaded.\n")

    print("[ 3/5 ] Building CommitBugClassifier (dual-head) ...")
    model = CommitBugClassifier(
        gpt_model=gpt,
        num_intent_classes=NUM_INTENT_CLASSES,
        num_severity_classes=NUM_SEVERITY_CLASSES,
    )
    freeze_backbone(model)
    model.to(device)
    print(f"       Trainable parameters: {count_trainable(model):,}\n")

    print("[ 4/5 ] Training ...")
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=0.1
    )

    start = time.time()
    results = train_classifier(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs,
        eval_freq=50,
        eval_iter=5,
    )
    elapsed = (time.time() - start) / 60
    print(f"\n       Training completed in {elapsed:.2f} minutes.\n")

    print("[ 5/5 ] Final evaluation on test set ...")
    test_intent_acc, test_severity_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"       Test intent   accuracy : {test_intent_acc*100:.2f}%")
    print(f"       Test severity accuracy : {test_severity_acc*100:.2f}%\n")

    torch.save(model.state_dict(), "commit_bug_classifier.pth")
    print("       Model saved -> commit_bug_classifier.pth\n")

    print("=" * 60)
    print("  INFERENCE DEMO")
    print("=" * 60)

    demo_commits = [
        "fix memory leak in redis connection pool",
        "patch SQL injection vulnerability in user login endpoint",
        "resolve race condition in async job queue",
        "update deployment pipeline missing env variable causing crash",
        "fix 403 on OAuth token refresh for mobile clients",
    ]

    if args.infer:
        demo_commits = [args.infer] + demo_commits

    for commit in demo_commits:
        result = classify_commit(commit, model, tokenizer, device, max_length=max_seq_len)
        print(f"\n  Commit  : \"{commit}\"")
        print(f"  Intent  : {result['intent']}  ({result['intent_confidence']*100:.1f}%)")
        print(f"  Severity: {result['severity']} ({result['severity_confidence']*100:.1f}%)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commit Bug Classifier")
    parser.add_argument("--data",          type=str,  default="msr2013-bug_dataset")
    parser.add_argument("--epochs",        type=int,  default=5)
    parser.add_argument("--batch-size",    type=int,  default=8)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--checkpoint",    type=str,  default="commit_bug_classifier.pth")
    parser.add_argument("--infer",         type=str,  default=None)
    args = parser.parse_args()
    main(args)