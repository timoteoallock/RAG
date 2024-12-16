import pandas as pd
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from answer_retrieval import query_database

model_name = "google/gemma-2b-it"
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  
).to(device)


# Define the new evaluation prompt (without reference answer, assessing quality based on criteria)
EVALUATION_PROMPT = """### Task Description:
You will be given a question, a context (if available), and a response to evaluate.
1. Assess the quality of the response based on the following criteria:
   - Correctness: Is the response factually correct based on the context, question and ground truth?
   - Completeness: Does the response cover all relevant points in answering the question?
   - Clarity: Is the response clear and easy to understand?
2. Provide a score between 1 and 5 for each criterion (1 being poor, 5 being excellent).
3. The output format should look as follows: 'Correctness: {{your score}}, Completeness: {{your score}}, Clarity: {{your score}} [RESULT] {{average score}}'
You must include '[RESULT]' in your output.

### Question:
{question}

### Context:
{context}

### Response:
{response}

## Ground Truth: 
{groundtruth}

### Feedback:
"""


def generate_llm_answer(question, context="", max_new_tokens=150):

    if isinstance(context, list):
        context = "\n".join(context)
    else:
        context = context if context else ""
    input_text = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"With the help of the provided context, answer the above question:"
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],  
        attention_mask=inputs["attention_mask"],  
        max_new_tokens=max_new_tokens  
    )
    print("Raw Output:", tokenizer.decode(outputs[0]))

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_response(question, response, groundtruth, context="", max_new_tokens=150):
    if isinstance(context, list):
        context_str = "\n".join(context)
    else:
        context_str = context if context else "No context available"

    eval_prompt = EVALUATION_PROMPT.format(
        question=question,
        response=response,
        groundtruth=groundtruth,
        context=context_str
    )
    
    inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=max_new_tokens  
    )
    
    evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        average_score = float(evaluation.split("[RESULT]")[-1].strip())
    except (ValueError, IndexError):
        average_score = None
    return evaluation, average_score

def evaluate_dataframe(df, collection_name="medical_qa_collection", persist_path=""):
    results_list = []
    plain_llm_scores = []
    rag_llm_scores = []
    
    for _, row in df.iterrows():
        question = row['question']
        answer = row['ground_truth']
        rag_retrieved_answers = query_database(question, collection_name, persist_path, rerank=True)
        rag_context = rag_retrieved_answers[0] if rag_retrieved_answers else "No relevant context found."
        
        rag_answer = generate_llm_answer(question, rag_context)
        rag_evaluation, rag_score = evaluate_response(question, rag_answer, answer, rag_context)
        
        plain_llm_answer = generate_llm_answer(question)
        plain_evaluation, plain_score = evaluate_response(question, plain_llm_answer, answer)
        
        plain_llm_scores.append(plain_score)
        rag_llm_scores.append(rag_score)
        
        results_list.append({
            'question': question,
            'rag_answer': rag_answer,
            'plain_llm_answer': plain_llm_answer,
            'rag_evaluation': rag_evaluation,
            'plain_evaluation': plain_evaluation,
            'rag_score': rag_score,
            'plain_score': plain_score
        })
    
    results_df = pd.DataFrame(results_list)
    avg_plain_score = sum([s for s in plain_llm_scores if s is not None]) / len([s for s in plain_llm_scores if s is not None])
    avg_rag_score = sum([s for s in rag_llm_scores if s is not None]) / len([s for s in rag_llm_scores if s is not None])
    print(f"Average Plain LLM Score: {avg_plain_score}")
    print(f"Average RAG Score: {avg_rag_score}")
    return results_df


if __name__ == "__main__":
    
    df = pd.read_csv("path\to\evaluation\dataset")

    evaluated_df = evaluate_dataframe(df)
    evaluated_df.to_csv('evaluated_results.csv', index=False)



