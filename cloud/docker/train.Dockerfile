FROM python:3.11-slim

WORKDIR /app
COPY evidence/cloud/src /app/src
COPY evidence/finetune/configs/run-a.yaml /app/configs/run-a.yaml
RUN pip install --no-cache-dir pyyaml peft transformers accelerate
CMD ["python", "src/train_peft.py", "--dry-run"]
