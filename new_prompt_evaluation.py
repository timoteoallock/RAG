import pandas as pd
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from answer_retrieval import query_database


model_name = "google/flan-t5-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

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


def generate_llm_answer(question, context=""):
    input_text = f" Answer the following question: {question}\n you can use the following context: {context}" if context else f"Question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_response_with_flant5(question, response, groundtruth, context=""):
    eval_prompt = EVALUATION_PROMPT.format(
        question=question, response=response, groundtruth = groundtruth, context=context if context else "No context available"
    )
    
    
    inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True).to(device)
    
    
    outputs = model.generate(**inputs)
    
    
    evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    try:
        average_score = float(evaluation.split("[RESULT]")[-1].strip())
    except ValueError:
        average_score = None  
    return evaluation, average_score

# Main evaluation function that processes the dataframe
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
        rag_evaluation, rag_score = evaluate_response_with_flant5(question, rag_answer,answer, rag_context)
        
        
        plain_llm_answer = generate_llm_answer(question)
        plain_evaluation, plain_score = evaluate_response_with_flant5(question, answer, plain_llm_answer)
        
        
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
    
    # Calculate and print average scores
    avg_plain_score = sum([score for score in plain_llm_scores if score is not None]) / len(plain_llm_scores)
    avg_rag_score = sum([score for score in rag_llm_scores if score is not None]) / len(rag_llm_scores)
    print(f"Average Plain LLM Score: {avg_plain_score}")
    print(f"Average RAG Score: {avg_rag_score}")
    
    return results_df


if __name__ == "__main__":
    
    df = pd.read_csv("C:\\Users\\albert_f\\machine_learning\\rag_project\\dataset")

    # Run evaluation
    evaluated_df = evaluate_dataframe(df)

    # Save results to CSV
    evaluated_df.to_csv('evaluated_results.csv', index=False)

