# evaluation.py

import json
from typing import Dict, Any, List

from openai import OpenAI
from pydantic import BaseModel


client = None  # Global variable to store the OpenAI client


class EvaluationResult(BaseModel):
    score: float
    reason: str


def initialize_client(api_key: str):
    global client
    client = OpenAI(api_key=api_key)


def evaluate_rag(data: Dict[str, Any], api_key: str, model: str) -> Dict[str, float]:

    initialize_client(api_key)

    query = data.get("query", {}).get("text", "")
    answer = data.get("generate", {}).get("answer", "")
    context = data.get("generate", {}).get("sources", [])
    ground_truth = data.get("ground_truth")  # Now it's None if not present

    results = {
        "Faithfulness": evaluate_faithfulness(query, answer, context, model),
        "Context Precision": evaluate_context_precision(query, answer, context, model),
        "Relevance": evaluate_relevance(query, answer, context, model),
        "Context Recall": evaluate_context_recall(answer, context, model),
        "Context Relevancy": evaluate_context_relevancy(query, context, model),
    }

    # Only include evaluations that require ground truth if it's available
    if ground_truth:
        results.update({
            "Context Entities Recall": evaluate_context_entities_recall(
                ground_truth, context, model
            ),
            "Answer Semantic Similarity": evaluate_answer_semantic_similarity(
                answer, ground_truth, model
            ),
            "Answer Correctness": evaluate_answer_correctness(answer, ground_truth, model),
        })

    return results


def generate_response(system_prompt: str, user_prompt: str, model: str) -> Dict:

    global client
    
    if client is None:
        raise ValueError("OpenAI client not initialized. Call initialize_client first.")

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format=EvaluationResult,
    )
    
    res = response.choices[0].message.parsed
    
    return {"score": res.score, "reason": res.reason}


def evaluate_faithfulness(
    query: str, answer: str, context: List[str], model: str
):
    system_prompt = "You are an impartial judge evaluating the faithfulness of an answer to a question based on given context. Analyze the answer's claims, determine if each claim can be inferred from the context, and calculate a faithfulness score. Be objective and thorough in your assessment."

    user_prompt = f"""
    Given:
    - Question: {query}
    - Answer: {answer}
    - Context: {context}

    Please follow these steps:

    1. Identify claims in the answer:
      a. Break down the answer into individual claims or statements.
      b. List each claim separately.
      c. Count the total number of claims.

    2. Analyze each claim:
      a. For each claim, determine if it can be inferred from the given context.
      b. Provide a brief explanation for your decision on each claim.
      c. Count the number of claims that can be inferred from the context.

    3. Calculate the faithfulness score:
      a. Use the formula: (Number of claims inferred from context) / (Total number of claims)
      b. Express the result as a decimal between 0 and 1.

    4. Provide a summary:
      a. State the faithfulness score.
      b. Summarize your evaluation, including:
          - Total number of claims in the answer
          - Number of claims that can be inferred from the context
          - Any notable observations about the answer's faithfulness to the context
      c. If applicable, highlight any claims in the answer that are not supported by the context.

    Please present your evaluation in a clear, structured format, showing your work for each step of the process.

    Example Output Format:

    1. Claims Identification:
      - Claim 1: [Text of claim]
      - Claim 2: [Text of claim]
      ...
      Total claims: [Number]

    2. Claims Analysis:
      - Claim 1: [Can/Cannot be inferred] - [Brief explanation]
      - Claim 2: [Can/Cannot be inferred] - [Brief explanation]
      ...
      Claims inferred from context: [Number]

    3. Faithfulness Score Calculation:
      [Number of inferred claims] / [Total claims] = [Score]

    4. Summary:
      Faithfulness Score: 
      <score>
      [Score]
      </score>
      [Summary text]
      [Unsupported claims, if any]

    Present your evaluation in a clear, structured format, showing your work for each step of the process.
    Include the final faithfulness evaluation with the score and reason.
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_relevance(
    query: str, answer: str, context: List[str], model: str
):
    system_prompt = "You are an AI assistant specialized in evaluating the relevancy of answers to questions. Your task is to generate artificial questions based on a given answer, compare them to the original question, and calculate an Answer Relevancy score. Use your language understanding to conceptualize semantic similarities without performing actual mathematical calculations."

    user_prompt = f"""
    1. Given:
      - Original Question: {query}
      - Generated Answer: {answer}
      - Context (if provided): {context}

    2. Task Overview:
      Your task is to generate artificial questions based on the answer, compare these to the original question, and calculate a relevancy score.

    3. Steps:

      a. Generate Questions:
          - Create 3 artificial questions that the given answer could be responding to.
          - These questions should be diverse and cover different aspects of the answer.
          - List these questions clearly.

      b. Conceptual Embedding and Similarity:
          - For each generated question and the original question, imagine you are creating a semantic embedding. This embedding would represent the meaning of the question in a high-dimensional space.
          - Conceptually compare each generated question to the original question. Consider how similar they are in meaning and intent.
          - Assign a similarity score between -1 and 1 for each comparison, where:
            * 1 indicates perfect similarity
            * 0 indicates no relation
            * -1 indicates opposite meanings
          - Explain your reasoning for each similarity score.

      c. Calculate Answer Relevancy:
          - Use the formula: Answer Relevancy = (Sum of similarity scores) / (Number of generated questions)
          - Show your calculation.

    4. Final Output:
      - List the original question and generated questions
      - Show similarity scores and explanations
      - Present the final Answer Relevancy score
      - Provide a brief interpretation of the score

    Remember, while you're conceptualizing embeddings and similarity, you're not actually generating numerical embeddings. Use your understanding of language and context to estimate similarity.

    Example Output Format:

    Original Question: [Original question text]

    Generated Questions:
    1. [Question 1]
    2. [Question 2]
    3. [Question 3]

    Similarity Scores:
    1. Score: [X.XX] - Explanation: [Your reasoning]
    2. Score: [X.XX] - Explanation: [Your reasoning]
    3. Score: [X.XX] - Explanation: [Your reasoning]

    Calculation:
    Answer Relevancy = (Score1 + Score2 + Score3) / 3 = <final_score>[Final Score]</final_score>

    Interpretation:
    [Brief interpretation of the score and what it means for the answer's relevancy]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_context_precision(
    query: str, answer: str, context: List[str], model: str
):
    system_prompt = "You are an AI assistant specialized in evaluating the precision of context chunks for given questions. Your task is to assess the relevance of each context chunk, calculate precision at various ranks, and compute an overall Context Precision score. Use your understanding of the question, ground truth, and contexts to make objective assessments and perform accurate calculations."

    user_prompt = f"""
    1. Given:
      - Question: {query}
      - Ground Truth: {answer}
      - Contexts: {context}

    2. Task Overview:
      Your task is to evaluate the relevance of each context chunk, calculate precision at each rank, and then compute the overall Context Precision score.

    3. Steps:

      a. Evaluate Relevance:
          - For each context chunk, determine if it's relevant to answering the question based on the ground truth.
          - Assign a relevance indicator (v_k) of 1 if relevant, 0 if not relevant.
          - List your decisions and briefly explain your reasoning for each.

      b. Calculate Precision at each rank (Precision@k):
          - For each rank k from 1 to K:
            * Count the number of relevant items up to and including rank k (true positives).
            * Calculate Precision@k = (true positives at k) / k
          - Show your calculations for each k.

      c. Calculate Context Precision@K:
          - Sum the products of (Precision@k * v_k) for all k from 1 to K.
          - Divide this sum by the total number of relevant items in the top K results.
          - Show your calculation.

    4. Final Output:
      - List the relevance decisions for each context chunk
      - Show Precision@k calculations for each k
      - Present the final Context Precision@K score
      - Provide a brief interpretation of the score

    Remember to be objective in your relevance assessments and precise in your calculations.

    Example Output Format:

    Question: [Question text]
    Ground Truth: [Ground truth text]

    Relevance Evaluations:
    1. Context Chunk 1: [Relevant/Not Relevant] - v_1 = [0/1] - Explanation: [Brief reasoning]
    2. Context Chunk 2: [Relevant/Not Relevant] - v_2 = [0/1] - Explanation: [Brief reasoning]
    ...
    K. Context Chunk K: [Relevant/Not Relevant] - v_K = [0/1] - Explanation: [Brief reasoning]

    Precision@k Calculations:
    Precision@1 = [calculation] = [result]
    Precision@2 = [calculation] = [result]
    ...
    Precision@K = [calculation] = [result]

    Context Precision@K Calculation:
    Sum of (Precision@k * v_k) = [calculation]
    Total number of relevant items = [number]
    Context Precision@K = [final calculation] = [Final Score]

    Interpretation:
    [Brief interpretation of the score and what it means for the context precision]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_context_relevancy(query: str, context: List[str], model: str):
    system_prompt = "You are an AI assistant specialized in evaluating the relevancy of retrieved context for given questions. Your task is to analyze individual sentences, determine their relevance to the question, and compute an overall Context Relevancy score. Use your understanding of the question and context to make objective assessments and perform accurate calculations."

    user_prompt = f"""
    1. Given:
      - Question: {query}
      - Retrieved Context: {context}

    2. Task Overview:
      Your task is to identify relevant sentences in the retrieved context, count them, and calculate the Context Relevancy score.

    3. Steps:

      a. Sentence Identification:
          - Break down the retrieved context into individual sentences.
          - Number each sentence for easy reference.

      b. Relevance Evaluation:
          - For each sentence, determine if it's relevant to answering the question.
          - Assign a relevance indicator of 1 if relevant, 0 if not relevant.
          - Briefly explain your reasoning for each decision.

      c. Calculate Context Relevancy:
          - Count the total number of sentences in the retrieved context.
          - Count the number of relevant sentences (|S|).
          - Calculate the Context Relevancy score using the formula:
            Context Relevancy = |S| / (Total number of sentences in retrieved context)
          - Show your calculation.

    4. Final Output:
      - List all sentences with their relevance decisions
      - Show the calculation of the Context Relevancy score
      - Present the final Context Relevancy score
      - Provide a brief interpretation of the score

    Remember to be objective in your relevance assessments and precise in your calculations.

    Example Output Format:

    Question: [Question text]

    Sentence Evaluation:
    1. [Sentence 1]: [Relevant/Not Relevant] - Explanation: [Brief reasoning]
    2. [Sentence 2]: [Relevant/Not Relevant] - Explanation: [Brief reasoning]
    ...
    N. [Sentence N]: [Relevant/Not Relevant] - Explanation: [Brief reasoning]

    Calculation:
    Total number of sentences: [N]
    Number of relevant sentences (|S|): [X]
    Context Relevancy = X / N = [Final Score]

    Interpretation:
    [Brief interpretation of the score and what it means for the context relevancy]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_context_recall(
    answer: str, context: List[str], model: str
):
    system_prompt = "You are an AI assistant specialized in evaluating the recall of retrieved context compared to ground truth answers. Your task is to analyze individual sentences from the ground truth, determine their attribution to the retrieved context, and compute an overall Context Recall score. Use your understanding of the ground truth and context to make objective assessments and perform accurate calculations."

    user_prompt = f"""
    1. Given:
      - Ground Truth Answer: {answer}
      - Retrieved Context: {context}

    2. Task Overview:
      Your task is to analyze each sentence in the ground truth answer, determine if it can be attributed to the retrieved context, and calculate the Context Recall score.

    3. Steps:

      a. Ground Truth Sentence Identification:
          - Break down the ground truth answer into individual sentences.
          - Number each sentence for easy reference.

      b. Attribution Evaluation:
          - For each ground truth sentence, determine if it can be attributed to (found in or inferred from) the retrieved context.
          - Assign an attribution indicator of 1 if attributable, 0 if not attributable.
          - Briefly explain your reasoning for each decision.

      c. Calculate Context Recall:
          - Count the total number of sentences in the ground truth answer.
          - Count the number of ground truth sentences that can be attributed to the context.
          - Calculate the Context Recall score using the formula:
            Context Recall = (Number of GT sentences attributed to context) / (Total number of sentences in GT)
          - Show your calculation.

    4. Final Output:
      - List all ground truth sentences with their attribution decisions
      - Show the calculation of the Context Recall score
      - Present the final Context Recall score
      - Provide a brief interpretation of the score

    Remember to be objective in your attribution assessments and precise in your calculations.

    Example Output Format:

    Ground Truth Sentence Evaluation:
    1. [GT Sentence 1]: [Attributable/Not Attributable] - Explanation: [Brief reasoning]
    2. [GT Sentence 2]: [Attributable/Not Attributable] - Explanation: [Brief reasoning]
    ...
    N. [GT Sentence N]: [Attributable/Not Attributable] - Explanation: [Brief reasoning]

    Calculation:
    Total number of GT sentences: [N]
    Number of GT sentences attributable to context: [X]
    Context Recall = X / N = [Final Score]

    Interpretation:
    [Brief interpretation of the score and what it means for the context recall]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_context_entities_recall(
    ground_truth: List[str], context: List[str], model: str
):
    system_prompt = "You are an AI assistant specialized in evaluating the recall of entities in retrieved context compared to ground truth. Your task is to identify entities in both the ground truth and context, compare these sets, and compute a Context Entities Recall score. Use your understanding of entity recognition to make thorough identifications and perform accurate calculations."

    user_prompt = f"""
    1. Given:
      - Ground Truth: {ground_truth}
      - Retrieved Context: {context}

    2. Task Overview:
      Your task is to identify entities in both the ground truth and the retrieved context, compare these sets, and calculate the Context Entities Recall score.

    3. Steps:

      a. Entity Identification:
          - Identify all entities in the ground truth. List them as set GE.
          - Identify all entities in the retrieved context. List them as set CE.
          - Entities may include named individuals, organizations, locations, dates, numerical facts, and other specific, identifiable information.

      b. Set Comparison:
          - Identify the entities that appear in both GE and CE (the intersection).
          - List these common entities.

      c. Calculate Context Entities Recall:
          - Count the number of entities in GE.
          - Count the number of entities in the intersection of GE and CE.
          - Calculate the Context Entities Recall score using the formula:
            Context Entities Recall = |GE ∩ CE| / |GE|
            (Where |GE ∩ CE| is the number of entities in the intersection, and |GE| is the total number of entities in the ground truth)
          - Show your calculation.

    4. Final Output:
      - List all entities found in the ground truth (GE)
      - List all entities found in the retrieved context (CE)
      - List the entities common to both (GE ∩ CE)
      - Show the calculation of the Context Entities Recall score
      - Present the final Context Entities Recall score
      - Provide a brief interpretation of the score

    Remember to be thorough in your entity identification and precise in your calculations.

    Example Output Format:

    Entities in Ground Truth (GE):
    [List of entities]

    Entities in Retrieved Context (CE):
    [List of entities]

    Common Entities (GE ∩ CE):
    [List of common entities]

    Calculation:
    |GE| (Total entities in ground truth): [Number]
    |GE ∩ CE| (Common entities): [Number]
    Context Entities Recall = |GE ∩ CE| / |GE| = [Final Score]

    Interpretation:
    [Brief interpretation of the score and what it means for the context entities recall]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_answer_semantic_similarity(
    answer: str, ground_truth: List[str], model: str
):
    system_prompt = "You are an AI assistant specialized in evaluating the semantic similarity between generated answers and ground truth answers. Your task is to analyze the content, structure, and meaning of both answers, and compute an Answer Semantic Similarity score. Use your understanding of language and semantics to make thorough comparisons and provide a justified similarity score."

    user_prompt = f"""
    1. Given:
      - Ground Truth Answer: {ground_truth}
      - Generated Answer: {answer}

    2. Task Overview:
      Your task is to compare the semantic meaning of the generated answer to the ground truth answer and assign a similarity score between 0 and 1, where 1 indicates perfect semantic similarity and 0 indicates no semantic similarity.

    3. Steps:

      a. Content Analysis:
          - Identify the main concepts, facts, and arguments present in both the ground truth and generated answer.
          - List these key elements for each answer.

      b. Structural Comparison:
          - Compare the organization and flow of ideas between the two answers.
          - Note any significant differences or similarities in structure.

      c. Semantic Evaluation:
          - Assess how well the generated answer captures the meaning and intent of the ground truth answer.
          - Consider factors such as:
            * Accuracy of information
            * Completeness of the response
            * Relevance of the content
            * Consistency in terminology and concepts
            * Depth of explanation

      d. Assign Similarity Score:
          - Based on your analysis, assign a similarity score between 0 and 1.
          - Provide a detailed justification for your score, referencing specific aspects of your analysis.

    4. Final Output:
      - List the key elements identified in both answers
      - Summarize your structural and semantic comparison
      - Present the final Answer Semantic Similarity score
      - Provide a detailed explanation of how you arrived at this score

    Remember to focus on the semantic meaning rather than exact wording, and be as objective as possible in your assessment.

    Example Output Format:

    Ground Truth Key Elements:
    [List of key elements]

    Generated Answer Key Elements:
    [List of key elements]

    Structural Comparison:
    [Summary of structural similarities and differences]

    Semantic Evaluation:
    [Detailed analysis of semantic similarity, addressing the factors mentioned]

    Answer Semantic Similarity Score: [Score between 0 and 1]

    Justification:
    [Detailed explanation of the score, referencing specific aspects of the analysis]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response


def evaluate_answer_correctness(
    answer: str, ground_truth: List[str], model: str
):
    system_prompt = "You are an AI assistant specialized in evaluating the correctness of generated answers compared to ground truth answers. Your task is to assess both semantic and factual similarity, combine these assessments into a weighted score, and optionally apply a threshold for binary classification. Use your understanding of language and facts to make thorough comparisons and provide justified scores."

    user_prompt = f"""
    1. Given:
      - Ground Truth Answer: {ground_truth}
      - Generated Answer: {answer}
      - Semantic Weight: 0.5
      - Factual Weight: 0.5
      - Threshold (optional): None

    2. Task Overview:
      Your task is to evaluate both the semantic and factual similarity between the generated answer and the ground truth, combine these scores using the given weights, and calculate an overall Answer Correctness score.

    3. Steps:

      a. Semantic Similarity Evaluation:
          - Assess how well the generated answer captures the meaning and intent of the ground truth answer.
          - Consider factors such as:
            * Consistency in terminology and concepts
            * Completeness of the response
            * Depth of explanation
          - Assign a semantic similarity score between 0 and 1.
          - Briefly justify your semantic similarity score.

      b. Factual Similarity Evaluation:
          - Identify key facts, figures, and claims in both answers.
          - Compare the accuracy of these elements between the generated answer and ground truth.
          - Consider factors such as:
            * Correctness of specific data points
            * Accuracy of statements and claims
            * Presence of all crucial facts from the ground truth
          - Assign a factual similarity score between 0 and 1.
          - Briefly justify your factual similarity score.

      c. Calculate Weighted Answer Correctness Score:
          - Use the formula: 
            Answer Correctness = (Semantic Weight * Semantic Score) + (Factual Weight * Factual Score)
          - Show your calculation.

      d. Apply Threshold (if provided):
          - If a threshold is given, convert the score to binary:
            * If Answer Correctness >= Threshold, set score to 1
            * If Answer Correctness < Threshold, set score to 0

    4. Final Output:
      - Present the Semantic Similarity score with justification
      - Present the Factual Similarity score with justification
      - Show the calculation of the weighted Answer Correctness score
      - If applicable, show the binary threshold conversion
      - Provide the final Answer Correctness score
      - Give a brief interpretation of what the score means for the answer's correctness

    Remember to be as objective as possible in your assessment and provide clear justifications for your scores.

    Example Output Format:

    Semantic Similarity Score: [Score between 0 and 1]
    Justification: [Brief explanation]

    Factual Similarity Score: [Score between 0 and 1]
    Justification: [Brief explanation]

    Weighted Answer Correctness Calculation:
    ([Semantic Weight] * [Semantic Score]) + ([Factual Weight] * [Factual Score]) = [Weighted Score]

    [If applicable] Threshold Application:
    Original Score: [Weighted Score]
    Threshold: [Threshold Value]
    Binary Score: [0 or 1]

    Final Answer Correctness Score: [Final Score]

    Interpretation:
    [Brief interpretation of the score and what it means for the answer's correctness]
    """

    response = generate_response(system_prompt, user_prompt, model)
    return response
