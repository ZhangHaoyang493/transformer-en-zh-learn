import argparse
from pathlib import Path

import torch


@torch.inference_mode()
def test(model, input_tensor, tokenizer, max_len=64):
    """Greedily decode one input of shape (1, src_len) with a model in eval mode.

    Return the generated token IDs, including EOS if it was generated.
    """
    if input_tensor.ndim != 2 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must have shape (1, src_len)")

    device = next(model.parameters(), input_tensor).device
    input_tensor = input_tensor.to(device=device, dtype=torch.long)
    decoder_ids = input_tensor.new_full((1, 1), tokenizer.bos_token_id)
    src_mask = input_tensor.ne(tokenizer.pad_token_id)[:, None, None, :]

    generated_ids = []
    for _ in range(max_len):
        tgt_len = decoder_ids.size(1)
        tgt_mask = torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device).tril()
        logits = model(input_tensor, decoder_ids, src_mask, tgt_mask[None, None])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_id.item()
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        decoder_ids = torch.cat((decoder_ids, next_id), dim=-1)

    return generated_ids


def main():
    parser = argparse.ArgumentParser(description="加载 checkpoint 进行英译中")
    parser.add_argument(
        "checkpoint", nargs="?", type=Path, help="权重文件路径，默认使用 ckpt/ 下步数最大的文件"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_dir = Path(__file__).resolve().parent / "ckpt"
        checkpoints = (
            path for path in checkpoint_dir.glob("step_*.pt")
            if path.stem.removeprefix("step_").isdecimal() and path.is_file()
        )
        checkpoint = max(
            checkpoints, key=lambda path: int(path.stem.removeprefix("step_")), default=None
        )
        if checkpoint is None:
            parser.error(f"在 {checkpoint_dir} 中找不到 checkpoint")
    if not checkpoint.is_file():
        parser.error(f"找不到 checkpoint 文件 {checkpoint}")

    from basicModel.encoder_decoder_arch import EncoderDecoderArch
    from dataset import get_EN2CN_tokenizer

    try:
        tokenizer = get_EN2CN_tokenizer(local_files_only=True)
    except OSError:
        tokenizer = get_EN2CN_tokenizer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型结构和 train.py 保持一致。
    model = EncoderDecoderArch(
        dim=16,
        src_vocab=len(tokenizer),
        tar_vocab=len(tokenizer),
        enc_n=2,
        dec_n=4,
        head_n=4,
        dila_w=1.66,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    del state
    model.to(device).eval()
    max_src_len = model.posEnc.pos_emb.size(1)

    print(f"已加载 {checkpoint.name}（{device}）。输入英文，输入 /quit 退出。")
    while True:
        try:
            text = input("英文> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text == "/quit":
            break
        if not text:
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_src_len)
        generated_ids = test(model, inputs["input_ids"], tokenizer)
        translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print("中文>", translation or "（模型未生成可显示的译文）")


if __name__ == "__main__":
    main()
