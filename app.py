pip install torch transformers accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = "tiiuae/Falcon3-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

EMERGENCY_KEYWORDS = {
    "abdominal tuberculosis", "accident- domestic", "accident- industrial", "accident- non vehicular",
    "accident- vehicular", "aids", "allergy ?drug", "altered mental status",
    "amputation", "animal bite/ insect bite", "assault- sexual", "assault-mental", "assault-physical",
    "asthma", "bleeding injuries", "blood related problem", "bone ? tuberculosis", "breathing difficulty",
    "burn (scald/fire/chemical)", "cancer", "cardiac related", "chest pain", "convulsion/ seizures",
    "diabetes / hyperglycemia", "hypoglycemia", "hypotension", "hypertension ( bp problem)",
    "electrocution ( thunder bolt)", "electrocution (electric shock)", "ent bleed", "fainting",
    "fall victim", "fracture", "jaundice", "labour/delivery related problem", "liver cirrhosis",
    "lung tuberculosis", "malaria", "poisoning ? food", "poisoning (insecticide/pesticide)",
    "pregnancy ? related problem", "pregnancy related/bleeding/miscarriage/htn/labour pain",
    "rectal bleeding", "renal calculi", "stroke", "suicidal tendency", "throat- cough-blood",
    "typhoid", "urine- blood (haematuria)", "urine- retention", "vomiting ?blood", "covid-19 related"
}

# Example user inputs
age = input("Enter age: ")
gender = input("Enter gender: ")
chief_complaint = input("Enter chief complaint: ").lower().strip()

# Emergency check
if any(keyword in chief_complaint for keyword in EMERGENCY_KEYWORDS):
    print("\nThis sounds like a possible emergency.")
    print("Please call 108 emergency services immediately.")
    print("Do not try to manage this alone at home.")
else:
    # Build prompt for LLM
    prompt = f"""
    You are a healthcare assistant.

    Patient Details:
    - Age: {age}
    - Gender: {gender}
    - Chief Complaint: {chief_complaint}

    Task:
    - Predict the most likely common cause of this complaint.
    - Do NOT provide any medicines, prescriptions, or clinical advice.
    - Give 2 to 3 bullet points.
    - Keep the language simple and clear.

    Answer:
    """

    response = qa_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]

    if "Answer:" in response:
        final_output = response.split("Answer:")[-1].strip()
    else:
        final_output = response.strip()

    print("\nMost Likely Causes:\n", final_output)
