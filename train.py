import argparse
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from dataset import *
from basicModel.encoder_decoder_arch import EncoderDecoderArch

def train_model(
    batch_size=16, max_len=128, epoch=10, lr=1e-4, *,
    ckpt_dir="ckpt", log_dir="runs/train", resume_from=None
):
    """Train up to the given total number of epochs, optionally from a checkpoint."""
    from torch.utils.tensorboard import SummaryWriter

    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"找不到 checkpoint 文件 {resume_from}")

    train_generator = torch.Generator()
    shuffle_seed = torch.randint(2**31, ()).item()
    train_dataset, val_dataset, tokenizer = get_EN2CN_Dataloader(
        batch_size=batch_size, max_len=max_len, generator=train_generator
    )

    bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id


    model = EncoderDecoderArch(dim=16,
                           src_vocab=len(tokenizer),
                           tar_vocab=len(tokenizer),
                           enc_n=2,
                           dec_n=4,
                           head_n=4,
                           dila_w=1.66)
    model = model.cuda(0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr
    )

    global_step = 0
    start_epoch = 0
    batches_to_skip = 0
    resumed_loss_sum = 0.0
    resumed_tokens = 0
    batches_per_epoch = len(train_dataset)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            if (
                checkpoint["batch_size"] != batch_size
                or checkpoint["batches_per_epoch"] != batches_per_epoch
            ):
                raise ValueError("续训需要使用相同的 batch_size 和每轮批次数。")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint["global_step"]
            start_epoch = checkpoint["epoch"]
            batches_to_skip = checkpoint["batches_completed"]
            shuffle_seed = checkpoint["shuffle_seed"]
            resumed_loss_sum = checkpoint.get("epoch_loss_sum", 0.0)
            resumed_tokens = checkpoint.get("epoch_tokens", 0)
            if batches_to_skip == batches_per_epoch:
                start_epoch += 1
                batches_to_skip = 0
        else:
            step = resume_from.stem.removeprefix("step_")
            if not resume_from.stem.startswith("step_") or not step.isdecimal():
                raise ValueError("旧版权重文件需要用 step_<步数>.pt 命名，以便恢复全局步数。")
            model.load_state_dict(checkpoint)
            global_step = int(step)
            start_epoch = global_step // batches_per_epoch
            print("旧版 checkpoint 仅包含模型权重，Adam 重新初始化，从当前 epoch 开头重新洗牌。")
        del checkpoint
        if start_epoch >= epoch:
            print(f"已达到目标 {epoch} 个 epoch，可用 --epochs 指定更大的总轮数。")
            return
        print(f"从第 {global_step} 步续训，进入第 {start_epoch + 1} 轮（已完成 {batches_to_skip} 批）。")

    ckpt_dir = Path(ckpt_dir)
    purge_step = global_step + 1 if resume_from is not None else None
    with SummaryWriter(log_dir=str(log_dir), purge_step=purge_step) as writer:
        for epo in range(start_epoch, epoch):
            train_generator.manual_seed(shuffle_seed + epo)
            batch_offset = batches_to_skip if epo == start_epoch else 0
            epoch_loss_sum = resumed_loss_sum if batch_offset else 0.0
            epoch_tokens = resumed_tokens if batch_offset else 0
            avg_loss = f"{epoch_loss_sum / epoch_tokens:.4f}" if epoch_tokens else "-"
            with tqdm(
                total=batches_per_epoch,
                initial=batch_offset,
                desc=f"Epoch {epo + 1}/{epoch}",
                unit="batch",
                postfix={"global_step": str(global_step), "epoch_avg_loss": avg_loss},
            ) as progress:
                remaining_batches = islice(train_dataset, batch_offset, None)
                for batch_index, tdata in enumerate(remaining_batches, start=batch_offset + 1):
                    input, label, attn_mask = tdata['input_ids'], tdata['labels'], tdata['attention_mask'] # attn_mask: B len  score: B, head_n, len, len
                    batch_tokens = label.ne(-100).sum().item()
                    input, label, attn_mask = input.cuda(0), label.cuda(0), attn_mask.cuda(0)

                    decoder_ids = torch.full_like(label, pad_id).cuda(0)
                    decoder_ids[:, 0] = bos_id
                    decoder_ids[:, 1:] = label[:, :-1]
                    decoder_ids.masked_fill_(decoder_ids==-100, pad_id)


                    ilen, tlen = input.shape[1], label.shape[1]
                    attn_mask = attn_mask.view(-1, 1, 1, ilen).bool()
                    # B head_n, len, dim
                    casual_mask = torch.tril(torch.ones(tlen, tlen))
                    casual_mask = casual_mask.view(1, 1, tlen, tlen).bool().cuda(0)


                    optimizer.zero_grad()
                    out = model(input, decoder_ids, attn_mask, casual_mask) # B, len, tar_cocab

                    loss = F.cross_entropy(
                        out.view(-1, len(tokenizer)),
                        label.view(-1),
                        ignore_index=-100
                    )

                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    batch_loss = loss.item()
                    if batch_tokens:
                        epoch_loss_sum += batch_loss * batch_tokens
                        epoch_tokens += batch_tokens
                        avg_loss = f"{epoch_loss_sum / epoch_tokens:.4f}"
                    progress.set_postfix(
                        global_step=str(global_step), epoch_avg_loss=avg_loss, refresh=False
                    )
                    progress.update()
                    writer.add_scalar("train/loss", batch_loss, global_step)

                    if global_step % 10_000 == 0 or batch_index == batches_per_epoch:
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = ckpt_dir / f"step_{global_step}.pt"
                        with TemporaryDirectory(dir=ckpt_dir) as temp_dir:
                            temp_path = Path(temp_dir) / checkpoint_path.name
                            torch.save(
                                {
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "global_step": global_step,
                                    "epoch": epo,
                                    "batches_completed": batch_index,
                                    "batches_per_epoch": batches_per_epoch,
                                    "batch_size": batch_size,
                                    "shuffle_seed": shuffle_seed,
                                    "epoch_loss_sum": epoch_loss_sum,
                                    "epoch_tokens": epoch_tokens,
                                },
                                temp_path,
                            )
                            temp_path.replace(checkpoint_path)


                    if global_step % 10_000 == 0:
                        val_loss_sum = 0.0
                        val_tokens = 0
                        model.eval()
                        try:
                            with torch.no_grad():
                                for vdata in val_dataset:
                                    input = vdata['input_ids'].cuda(0)
                                    label = vdata['labels'].cuda(0)
                                    attn_mask = vdata['attention_mask'].cuda(0)

                                    decoder_ids = torch.full_like(label, pad_id)
                                    decoder_ids[:, 0] = bos_id
                                    decoder_ids[:, 1:] = label[:, :-1]
                                    decoder_ids.masked_fill_(decoder_ids == -100, pad_id)

                                    ilen, tlen = input.shape[1], label.shape[1]
                                    attn_mask = attn_mask.view(-1, 1, 1, ilen).bool()
                                    casual_mask = torch.tril(torch.ones(tlen, tlen))
                                    casual_mask = casual_mask.view(1, 1, tlen, tlen).bool().cuda(0)
                                    out = model(input, decoder_ids, attn_mask, casual_mask)

                                    val_loss_sum += F.cross_entropy(
                                        out.reshape(-1, out.size(-1)),
                                        label.reshape(-1),
                                        ignore_index=-100,
                                        reduction="sum",
                                    ).item()
                                    val_tokens += label.ne(-100).sum().item()
                        finally:
                            model.train()

                        if val_tokens:
                            writer.add_scalar("val/loss", val_loss_sum / val_tokens, global_step)










if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练英中翻译模型")
    parser.add_argument("--resume", type=Path, help="从指定的 checkpoint 续训")
    parser.add_argument("--epochs", type=int, default=10, help="目标总轮数，含已训练轮次")
    args = parser.parse_args()

    train_model(
        batch_size=8,
        max_len=64,
        epoch=args.epochs,
        resume_from=args.resume,
        lr=1e-3
    )
