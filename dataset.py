from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader


def get_EN2CN_tokenizer(*, local_files_only=False):
    tokenizer_dir = snapshot_download(
        repo_id="facebook/nllb-200-distilled-600M",
        local_files_only=local_files_only,
        allow_patterns=[
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )

    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        src_lang="eng_Latn",
        tgt_lang="zho_Hans",
    )


def get_EN2CN_Dataloader(batch_size=16, max_len=128, *, generator=None):
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")
    tokenizer = get_EN2CN_tokenizer()

    def tokenize_batch(batch):
        pairs = batch["translation"]
        return tokenizer(
            [pair["en"] for pair in pairs],
            text_target=[pair["zh"] for pair in pairs],
            truncation=True,
            max_length=max_len,
        )

    train = dataset["train"]
    test = dataset["test"]

    tokenized_train = train.map(
        tokenize_batch,
        batched=True,
        remove_columns=train.column_names,
    )

    tokenized_test = test.map(
        tokenize_batch,
        batched=True,
        remove_columns=test.column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )

    test_loader = DataLoader(
        tokenized_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, test_loader, tokenizer


if __name__ == '__main__':
    train_loader, test_loader, tokenizer = get_EN2CN_Dataloader(batch_size=4, max_len=16)
    print(tokenizer.bos_token, tokenizer.bos_token_id)
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.pad_token_id)

    for batch in train_loader:
        print(batch['input_ids'][0])
        print(batch['labels'][0])
        print(batch['attention_mask'][0])
        break
