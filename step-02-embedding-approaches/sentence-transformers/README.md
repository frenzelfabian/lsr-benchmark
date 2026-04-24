# Sentence Transformers Embedding Approach

This approach uses [Sentence Transformers](https://www.sbert.net/) to embed queries and documents.

## Usage

```bash
python sentence_transformers_embeddings.py \
    --dataset <path-to-dataset> \
    --model <model-name> \
    --batch_size 32 \
    --output <output-dir>
```

## Run Unit Tests

```bash
pip install pytest sentence-transformers
python -m pytest test_embeddings.py -v
```

## Submission

Code submission to tira via (remove the --dry-run for upload):

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id sentence-transformers \
    --dataset tiny-example-20251002_0-training \
    --command '/sentence_transformers_embeddings.py --dataset $inputDataset --output $outputDir --model sentence-transformers/all-MiniLM-L6-v2' \
    --mount-hf-model sentence-transformers/all-MiniLM-L6-v2 \
    --dry-run
```

