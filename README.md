# ConformalLLM
## Extending Conformal Prediction to LLMs 

## Code Organization
conformal_llm_scores.py contains the python script for classification using 1-shot question prompts. It outputs three files
1) The softmax scores corresponding to each subjects for each of the 10 prompts
2) The accuracy for each subject prompt for mmlu-based 1-shot question as a dictionary where the key is the subject name and value is a list containing accuracy for each of the 10 prompts.
3) The accuracy for each subject prompt for gpt4-based 1-shot question as a dictionary where the key is the subject name and value is a list containing accuracy for each of the 10 prompts.

In conformal.ipynb, we have results for all conformal prediction experiments and gpt4 vs mmlu based prompt comparison. It requires the three files outputted by conformal_llm_scores.py to work.

If you would like to run the experiments from scratch, apply for LLaMA [https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform](aceess here) and then use the hugging face version of LLaMA by converting original LLaMA weights to hugging face version [https://huggingface.co/docs/transformers/main/model_doc/llama](refer here for instructions) and then run the conformal_llm_scores.py script.


