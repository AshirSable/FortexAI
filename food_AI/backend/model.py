from typing import List, Union, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import SYSTEM_PROMPT

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully.")


def generate_response(chat_input: Union[str, List[Dict[str, str]]]) -> str:
    """
    Accepts either:
    - A single string message (e.g. from direct CLI testing)
    - A list of turn dicts: [{'role': 'user'|'assistant', 'content': '...'}]
    """
    formatted_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip()
        }
    ]

    if isinstance(chat_input, list):
        for msg in chat_input:
            # Ensure the system prompt isn't duplicated if already present
            if msg.get("role") != "system":
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
    else:
        formatted_messages.append({
            "role": "user",
            "content": str(chat_input)
        })

    # Prepare inputs using the tokenizer's chat template
    inputs = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            temperature=0.2,       # Lower temperature stops meta commentary & rambling
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1, # Prevents repetitive loops
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice out newly generated tokens only
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


if __name__ == "__main__":
    conversation_history = []
    print("\nCraveAI Test CLI started. Type 'exit' to quit.\n")

    while True:
        user_message = input("Customer: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            break
        if not user_message:
            continue

        # Append user turn to local test history
        conversation_history.append({"role": "user", "content": user_message})

        # Generate response passing full conversation
        bot_response = generate_response(conversation_history)
        print(f"\nFoodAssist: {bot_response}\n")

        # Append assistant turn to history
        conversation_history.append({"role": "assistant", "content": bot_response})