import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME    = os.getenv("MODEL_NAME",    "facebook/opt-125m")
DATASET_PATH  = os.getenv("DATASET_PATH",  "dataset/training_data.jsonl")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",    "output")
ADAPTER_DIR   = os.getenv("ADAPTER_DIR",   "lora_adapter")
EPOCHS        = int(os.getenv("EPOCHS",    "3"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE","2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
LORA_RANK     = int(os.getenv("LORA_RANK", "16"))
LORA_ALPHA    = int(os.getenv("LORA_ALPHA","32"))

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Device : " + DEVICE)
print("[INFO] Model  : " + MODEL_NAME)

# ── load dataset ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

print("[INFO] Loading dataset from " + DATASET_PATH + " ...")
raw_data = load_jsonl(DATASET_PATH)

formatted = []
for d in raw_data:
    instruction = d["instruction"]
    inp         = d.get("input", "").strip()
    output      = d["output"]
    if inp:
        text = "### Instruction:\n" + instruction + "\n\n### Input:\n" + inp + "\n\n### Response:\n" + output
    else:
        text = "### Instruction:\n" + instruction + "\n\n### Response:\n" + output
    formatted.append({"text": text})

train_dataset = Dataset.from_list(formatted)
print("[INFO] Dataset size: " + str(len(train_dataset)) + " examples")

# ── tokenizer ─────────────────────────────────────────────────────────────────
print("[INFO] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── model ─────────────────────────────────────────────────────────────────────
print("[INFO] Loading model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
model = model.float()

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── loss logger ───────────────────────────────────────────────────────────────
class LossLogger(TrainerCallback):
    def __init__(self):
        self.steps  = []
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])

loss_logger = LossLogger()

# ── training args ─────────────────────────────────────────────────────────────
model.config.use_cache = False

training_args = SFTConfig(
    output_dir                  = ADAPTER_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 4,
    learning_rate               = LEARNING_RATE,
    fp16                        = False,
    logging_steps               = 5,
    save_strategy               = "epoch",
    dataset_text_field          = "text",
    report_to                   = "none",
    optim                       = "adamw_torch",
    use_cpu                     = True,

)

trainer = SFTTrainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    processing_class = tokenizer,
    callbacks     = [loss_logger],
)

# ── train ─────────────────────────────────────────────────────────────────────
print("[INFO] Training started ...")
trainer.train()
print("[INFO] Training complete.")

# ── save adapter ──────────────────────────────────────────────────────────────
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("[INFO] Adapter saved to " + ADAPTER_DIR)

# ── save loss curve ───────────────────────────────────────────────────────────
loss_png = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.figure(figsize=(10, 5))
plt.plot(loss_logger.steps, loss_logger.losses,
         "b-o", markersize=3, linewidth=1.5, label="Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("LoRA Fine-Tuning Loss - " + MODEL_NAME)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_png, dpi=150)
plt.close()
print("[INFO] Loss curve saved -> " + loss_png)
print("[DONE] All artefacts saved successfully.")
