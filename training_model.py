import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

#Load the pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# adding padding token
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv("C:\Users\Wa Ode Zalmawati\Desktop\AIFellowshipWeek4\cicd_handson\Week3NikeProductDescriptionsGenerator.csv")

descriptions = df['Product Description'].tolist()

#Tokenize
# tokenize product descriptions
def preprocess(desc):
  encodings = tokenizer(desc, truncation=True, padding=True)
  # return tokenizer(descriptions, return_tensors="pt", truncation=True, padding=True)
  return Dataset.from_dict(encodings)

train_dataset = preprocess(descriptions)

#Initiate data collator 
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# load model 
model = GPT2LMHeadModel.from_pretrained("gpt2")

# set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

# initialize Trainer for fine-tuning model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# fine-tune model
trainer.train()