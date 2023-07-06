import prompt_questions as p
import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from collections import defaultdict
import pickle

# List of task we consider
task_list = ['college_computer_science', 'formal_logic', 'high_school_computer_science',
             'computer_security', 'machine_learning',

             'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
             'college_medicine', 'professional_medicine',

             'business_ethics', 'professional_accounting', 'public_relations',
             'management', 'marketing'
             ]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def modify_task_data(task_data, token_limit, max_size_prompt_len):
    '''
    task_data: load_dataset('lukaemon/mmlu', subject_name), i.e., comes from mmlu subject
    token_limit: the maximum sized token used in forward pass (some questions are too large and thus
    are difficult to fit into memory given we use a single A100, thus we keep a token_limit of 1500 tokens.)
    max_size_prompt_len: Since we use 10 different prompts for one question which all differ in their one-shot
    question, the number of total questions may become different for each prompt. Thus we chose the
    max_size_prompt_len, which is the largest of 10 prompts, to remove questions that exceed token_limit,
    This results in same count of questions across all 10 prompts.

    Returns task_data with questions exceeding (token_limit-max_size_prompt_len) length tokens removed.
    '''
    new_task_data = {
        'train': defaultdict(list),
        'validation': defaultdict(list),
        'test': defaultdict(list),
    }
    for split in new_task_data.keys():
        for i in range(len(task_data[split])):
            q = task_data[split]['input'][i]
            a = task_data[split]['A'][i]
            b = task_data[split]['B'][i]
            c = task_data[split]['C'][i]
            d = task_data[split]['D'][i]
            target = task_data[split]['target'][i]
            if len(q) + max(map(len, [a, b, c, d])) + max_size_prompt_len < token_limit:
                new_task_data[split]['input'].append(q)
                new_task_data[split]['A'].append(a)
                new_task_data[split]['B'].append(b)
                new_task_data[split]['C'].append(c)
                new_task_data[split]['D'].append(d)
                new_task_data[split]['target'].append(target)
    return new_task_data


def get_prompt(task_data, task, question_num=0, prompt_q=None):
    '''
    task_data:
    Question num specifies which question will be used as prompt.
    If prompt_q is provided, it is used as 1-shot prompt question. This
    corresponds to GPT-4 based question prompts that we created. Else, we
    select question corresponding to question_num from the MMLU itself to
    generate the prompt. We select prompt from test set in this case,
    since train set is very small sometime and may not have 10 samples.
    We use 10 different prompts and take avergae over them to estimate
    performance on a subject. The function returns the 1-shot question prompt.
    '''

    if prompt_q is None:
        prompt_set = 'test'
        if question_num > len(task_data['test']['input']) - 1:
            print('prompt question id exceeds the length of test set')
            print('selecting last question of the test set')
            question_num = len(task_data['test']['input']) - 1
        prompt_add = f'This is a question from {task.replace("_", " ")}.\n'
        prompt_add += f"{task_data[prompt_set]['input'][question_num]}\n"
        for letter in ['A', 'B', 'C', 'D']:
            prompt_add += '    ' + letter + '. ' + task_data[prompt_set][letter][question_num] + '\n'
        prompt_add += f"The correct answer is option: {task_data[prompt_set]['target'][question_num]}\n"
    else:
        prompt_add = f'This is a question from {task.replace("_", " ")}.'
        prompt_add += prompt_q
        prompt_add += '\n'
    prompt_add += f"You are the world's best expert in {task.replace('_', ' ')}. "
    prompt_add += '''Reason step-by-step and answer the following question. '''
    return prompt_add


def get_question_dict(task_data, prompt_add, prompt_q_id=None):
    '''
    task_data: The task_data obtained after passing original mmlu dataset to modify_task_data
    prompt_add: prompt obtained from function get_prompt (either GPT-4 based or MMLU based question
    promots)
    prompt_q_id: The question_id from test set in MMLU that was used to create prompt. If prompt
    was from GPT-4 based question this is None. Else an integer specifying the question number.
    We remove this question num from dataset since it is part of the prompt itself.

    Returns:
    questions - containing a list of dictionary where each dictionary is a (key value) pair where
    each key is one of the option choices and each value is complete prompt+question+option string. The
    last token in the value string of the dictionary is same as key.

    answers - containing the list of answer key for each question


    see the sample question element

{'A': "This is a question from college computer science.\nAn integer c is a common divisor of two integers
x and y if and only if c is a divisor of x and c is a divisor of y. Which of the following sets of integers
could possibly be the set of all common divisors of two integers?\n (A) {-6,-2, -1, 1, 2, 6}\n (B) {-6, -2,
-1, 0, 1, 2, 6}\n (C) {-6, -3, -2, -1, 1, 2, 3, 6}\n (D) {-6, -3, -2, -1, 0, 1, 2, 3, 6}\nThe correct answer
is option C.\nYou are the world's best expert in college computer science. Reason step-by-step and answer the
following question. The Singleton design pattern is used to guarantee that only a single instance of a class
may be instantiated. Which of the following is (are) true of this design pattern?\nI. The Singleton class has
a static factory method to provide its instance.\nII. The Singleton class can be a subclass of another class.
\nIII. The Singleton class has a private constructor.\n(A) I only (B) II only (C) III only (D) I, II, and III
\nThe correct answer is option: A", 'B': "This is a question from college computer science.\nAn integer c is
a common divisor of two integers x and y if and only if c is a divisor of x and c is a divisor of y. Which of
the following sets of integers could possibly be the set of all common divisors of two integers?\n (A) {-6,-2,
-1, 1, 2, 6}\n (B) {-6, -2, -1, 0, 1, 2, 6}\n (C) {-6, -3, -2, -1, 1, 2, 3, 6}\n (D) {-6, -3, -2, -1, 0, 1, 2,
3, 6}\nThe correct answer is option C.\nYou are the world's best expert in college computer science. Reason
step-by-step and answer the following question. The Singleton design pattern is used to guarantee that only a
single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?
\nI. The Singleton class has a static factory method to provide its instance.\nII. The Singleton class can be
a subclass of another class.\nIII. The Singleton class has a private constructor.\n(A) I only (B) II only (C)
III only (D) I, II, and III \nThe correct answer is option: B", 'C': "This is a question from college computer
science.\nAn integer c is a common divisor of two integers x and y if and only if c is a divisor of x and c is
a divisor of y. Which of the following sets of integers could possibly be the set of all common divisors of two
integers?\n (A) {-6,-2, -1, 1, 2, 6}\n (B) {-6, -2, -1, 0, 1, 2, 6}\n (C) {-6, -3, -2, -1, 1, 2, 3, 6}\n (D)
{-6, -3, -2, -1, 0, 1, 2, 3, 6}\nThe correct answer is option C.\nYou are the world's best expert in college
computer science. Reason step-by-step and answer the following question. The Singleton design pattern is used
to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true
of this design pattern?\nI. The Singleton class has a static factory method to provide its instance.\nII. The
Singleton class can be a subclass of another class.\nIII. The Singleton class has a private constructor.\n(A)
I only (B) II only (C) III only (D) I, II, and III \nThe correct answer is option: C", 'D': "This is a question
from college computer science.\nAn integer c is a common divisor of two integers x and y if and only if c is a
divisor of x and c is a divisor of y. Which of the following sets of integers could possibly be the set of all
common divisors of two integers?\n (A) {-6,-2, -1, 1, 2, 6}\n (B) {-6, -2, -1, 0, 1, 2, 6}\n (C) {-6, -3, -2, -1,
1, 2, 3, 6}\n (D) {-6, -3, -2, -1, 0, 1, 2, 3, 6}\nThe correct answer is option C.\nYou are the world's best
expert in college computer science. Reason step-by-step and answer the following question. The Singleton design
pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following
is (are) true of this design pattern?\nI. The Singleton class has a static factory method to provide its
instance.\nII. The Singleton class can be a subclass of another class.\nIII. The Singleton class has a private
constructor.\n(A) I only (B) II only (C) III only (D) I, II, and III \nThe correct answer is option: D"}

    '''
    questions = []
    answers = []
    splits = ['train', 'validation', 'test']
    if prompt_q_id is not None:
        print(f'Excluding test set question no {prompt_q_id} from dataset')

    for split in splits:
        if split == 'train':
            start = 1  # In at least one subject, we found first train question to be unrelated to subject,
            # that's why we remove question 1.
        else:
            start = 0
        for i in range(start, len(task_data[split]['input'])):
            if split == 'test' and prompt_q_id is not None:
                if i == prompt_q_id:
                    # Don't add prompt question to the dataset
                    continue
            question_dict = {}
            # prompt_add = 'You know everything about college medicine. Answer this multiple now. Question: \n'
            prompt_q = prompt_add + task_data[split]['input'][i] + '\n'
            # prompt_q = mmlu_prompt[task] + "\n\n" + task_data['test'][i]['input'] + '\n'
            for letter in ['A', 'B', 'C', 'D']:
                prompt_q += '(' + letter + ') ' + task_data[split][letter][i] + ' '
            # prompt_q += "\nA: Let's think step by step."
            prompt_q += "\nThe correct answer is option: "
            for letter in ['A', 'B', 'C', 'D']:
                question_dict[letter] = prompt_q + letter
            questions.append(question_dict)
            answers.append(task_data[split]['target'][i])
    return questions, answers


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    '''
    Takes model, tokenizer and input_texts corresponding to each of the choices
    to do a forward pass through the model.
    Returns log-softmax scores as a list of tuples, where first element of tuple
    contains the option choice and second contains the corresponding log-softmax
    score. The list has size four corresponding to the four options.
    '''
    all_outputs = []
    all_input_ids = []
    for text in input_texts:
        input_ids = tokenizer(text, padding=True, return_tensors="pt").input_ids.to("cuda")
        outputs = model(input_ids)
        logits = outputs.logits.detach().cpu()
        all_outputs.append(logits)
        all_input_ids.append(input_ids.detach().cpu())
        del outputs, input_ids
        torch.cuda.empty_cache()

    all_outputs = torch.concat(all_outputs, 0)[:, -2:-1, :]  # We take the logit corresponding to the option token
    all_input_ids = torch.concat(all_input_ids, 0)[:, -1:]  # We also include the token id for the options
    probs = torch.log_softmax(all_outputs.float(), dim=-1).detach().cpu()  # Log softmax scores
    torch.cuda.empty_cache()

    gen_probs = torch.gather(probs, 2, all_input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(all_input_ids[:, 0], gen_probs[:, 0]):
        batch.append((tokenizer.decode(input_sentence), input_probs.item()))
    return batch


def softmax(logits):
    '''
    converts log-softmax scores to probablities.
    '''
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp_logits
    return probabilities


def extract_answer(batch):
    '''
    converts the batch of option, log-softmax score tuples to option, probablity tuples
    '''
    probabilities = softmax(np.array([answer[-1] for answer in batch]))

    output_with_probabilities = [(batch[i][0], probabilities[i]) for i in range(len(batch))]
    return output_with_probabilities


def average_question_predictions(prediction_list):
    '''
    Calculates the average of the probability for question-option pairs by avergaing the
    probability across prompts.
    '''
    num_seeds = len(prediction_list)  # Number of random seeds (or runs)
    average_list = []  # List to store the average predictions for each question

    # Iterate through each question
    for question_idx in range(len(prediction_list[0])):
        # Initialize a dictionary to store the sums of probabilities for each option
        option_sums = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # Iterate through each random seed
        for seed_idx in range(num_seeds):
            # Iterate through each option and its probability for the current question and seed
            for option, value in prediction_list[seed_idx][question_idx]:
                # Add the probability to the corresponding option sum
                option_sums[option] += value

        # Calculate the average probability for each option and store them as tuples
        option_averages = [(key, value / num_seeds) for key, value in option_sums.items()]
        # Add the average probabilities for the current question to the list
        average_list.append(option_averages)

    return average_list


def accuracy(predicted_probs, correct_answers):
    '''
    Given predicted probability for each question-option pairs and correct answer for that question,
    returns the accuracy.
    '''
    total_count = len(correct_answers)
    assert len(correct_answers) == len(predicted_probs)
    correct_count = 0

    for i in range(total_count):
        # Find the answer with the maximum probability for this example
        max_prob_answer = max(predicted_probs[i], key=lambda x: x[1])[0].strip()
        # print(max_prob_answer, correct_answers[i])
        # Compare the predicted answer with the correct answer
        if correct_answers[i] == max_prob_answer:
            correct_count += 1.0

    return correct_count / total_count


def get_max_size_prompt_len(task_data, task, n=10, max_allowed_prompt_len=700):
    '''
    get the size of maximum length prompt out of all n prompts considered.
    '''
    max_len = 0
    i = 0
    prompt_question_ids = []
    while len(prompt_question_ids) < n:
        prompt_add = get_prompt(task_data, task=task, question_num=i)
        prompt_len = len(prompt_add)

        if prompt_len > max_allowed_prompt_len:
            i += 1
            continue
        else:
            prompt_question_ids.append(i)
            i += 1

        if prompt_len > max_len:
            max_len = prompt_len
    return max_len, prompt_question_ids


def get_acc_index(preds, answers):
    '''
    Takes saved preds and answers and returns accuracy
    '''
    correct = 0
    for i in range(len(preds)):
        if preds[i].index(max(preds[i])) == answers[i]:
            correct += 1
    acc = correct / len(answers)
    return acc




token_limit = 1500  # Maximum size of tokens used in forward pass.
n = 10 # number of different MMLU based prompts used.
task_list = task_list

max_size_prompt_len_dict = {}
prompt_question_ids_dict = {}
for subject_name in task_list:
    task_data = load_dataset('lukaemon/mmlu', subject_name)
    max_len, prompt_question_ids = get_max_size_prompt_len(task_data, subject_name, n=n,
                                                          max_allowed_prompt_len=700)
    max_size_prompt_len_dict[subject_name] = max_len
    prompt_question_ids_dict[subject_name] = prompt_question_ids

save_dir = './llama_hf_13b'  # Model Directory

tokenizer = LlamaTokenizer.from_pretrained(save_dir, low_cpu_mem_usage=True)
model = LlamaForCausalLM.from_pretrained(save_dir, low_cpu_mem_usage=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.half().cuda()

# Get prediction for subjects with MMLU based prompts

acc_dicts = {}

for subject_name in task_list:
    task_data = load_dataset('lukaemon/mmlu', subject_name)
    new_task_data = modify_task_data(task_data, token_limit, max_size_prompt_len_dict[subject_name])

    acc_dicts[subject_name] = []
    print(f'generating predictions for the subject {subject_name}')
    for j, question_num in enumerate(prompt_question_ids_dict[subject_name]):
        preds = []
        targets = []
        print(f'Running experiments with test set question_id {question_num}')
        prompt_add = get_prompt(task_data, task=subject_name, question_num=question_num, prompt_q=None)
        if j % 5 == 0:
            print(prompt_add)
        questions, answers = get_question_dict(new_task_data, prompt_q_id=question_num, prompt_add=prompt_add)
        for i, (question, answer) in enumerate(zip(questions, answers)):
            batch = to_tokens_and_logprobs(model, tokenizer, [v for v in question.values()])
            torch.cuda.empty_cache()
            preds.append(extract_answer(batch))
            targets.append(answer)
        print(f'Predictions Generated for {subject_name} for iteration {j}')
        print('Calculating accuracy')
        acc = round(accuracy(preds, targets), 3)
        acc_dicts[subject_name].append(acc)
        print(f'Accuracy on {subject_name} for iteration {j} is {acc:.2f} ')
    print('*****************************************************************************************')
    print(f'calculating average accuracy on {subject_name}')
    print(f'Average accuracy on {subject_name} is {np.mean(np.array(acc_dicts[subject_name])):.3f}')
    with open("accuracy_mmlu_prompts_10.pkl", "wb") as f:
        pickle.dump(acc_dicts, f)


# Import GPT-4 based question prompts
prompt_list = [p.prompt_q_list_college_cs, p.prompt_q_list_formal_logic, p.prompt_q_list_high_school_cs,
               p.prompt_q_list_computer_security, p.prompt_q_list_machine_learning,

               p.prompt_q_list_clinical_knowledge, p.prompt_q_list_high_school_bio, p.prompt_q_list_anatomy,
               p.promtp_q_list_college_chemistry, p.prompt_q_list_college_medicine,
               p.prompt_q_list_professional_medicine,

               p.prompt_q_list_business_ethics, p.prompt_q_list_professional_accounting, p.prompt_q_list_pr,
               p.prompt_q_list_management, p.prompt_q_list_marketing
               ]


prompt_list = prompt_list

def get_predictions_over_n_runs(task_data, prompt_q_list, task):
    '''
    Takes into input mmlu dataset for a subject and list of GPT-4 based prompts for that subject
    Returns probablity scores (as list of list) for the mmlu questions for each options (A, B, C, D)
    and for each prompt along with the true answers along with the average accuracy over n runs.
    '''
    predictions_list = []
    acc_list = []

    for j, prompt_q in enumerate(prompt_q_list):
        prompt_add = get_prompt(task_data, task=task, prompt_q=prompt_q)
        if j % 5 == 0:
            print(prompt_add)
        questions, solution_answers = get_question_dict(task_data, prompt_add=prompt_add)
        predictions = []
        targets = []
        for (question, answer) in zip(questions, solution_answers):
            batch = to_tokens_and_logprobs(model, tokenizer, [v for v in question.values()])
            torch.cuda.empty_cache()
            predictions.append(extract_answer(batch))
            targets.append(answer)
        acc = round(accuracy(predictions, targets), 3)
        print(f'Accuracy on {task} for iteration {j} is {acc:.2f} ')
        acc_list.append(acc)
        predictions_list.append(predictions)
    return predictions_list, solution_answers, acc_list

def get_prediction_list(subject_name, prompt_list, token_limit=1500):
    '''
    Runs the get_predictions_over_n_runs function for a specific subject after removing questions
    that exceed the token limits.
    '''
    max_size_prompt = np.max(np.array([len(x) for x in prompt_list]))
    task_data = load_dataset('lukaemon/mmlu', subject_name)
    task_data_modified = modify_task_data(task_data, token_limit=token_limit,
                                          max_size_prompt_len=max_size_prompt)
    prediction_lists, solution_answers, avg_acc = get_predictions_over_n_runs(task_data_modified,
                                                                     prompt_list, subject_name)
    return prediction_lists, solution_answers, avg_acc

# Get predictions for each subject using GPT-4 based prompts

acc_dicts_mmlu = {}
for task, prompt in zip(task_list, prompt_list):
    prediction_lists, solution_answers, acc_list = get_prediction_list(task, prompt, token_limit)
    avg_acc = np.mean(np.array(acc_list))
    print('*****************************************************************************************')
    print(f'calculating average accuracy on {task}')
    print(f'Average accuracy on {task} is {avg_acc:.3f}')
    acc_dicts_mmlu[task] = acc_list
    with open("accuracy_gpt_prompts_10.pkl", "wb") as f:
        pickle.dump(acc_dicts_mmlu, f)
    scores = np.array([[[a[1] for a in p] for p in predictions] for predictions in prediction_lists])

    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    targets = np.array(list(map(lambda x: answer_map[x], solution_answers)))
    np.save(f'{task}_scores.npy', scores)
    np.save(f'{task}_targets.npy', targets)


