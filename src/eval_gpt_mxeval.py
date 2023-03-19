import openai
from mxeval.evaluation import (
  get_execute_function,
  estimate_pass_at_k,
)
from mxeval.data import get_data, get_examples, get_supported_langs
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
import numpy as np
import time
import argparse
## YOUR API KEY HERE
openai.api_key_path = ".openai_key"


def get_args():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                          choices=["gpt-3.5-turbo", "gpt-4"])
  arg_parser.add_argument("--dataset", type=str, default="multi-humaneval",
                          choices=["multi-humaneval", "mbxp", "mathqa-x"]
                          )
  arg_parser.add_argument("--num_shots", type=int, default=1,
                          )
  arg_parser.add_argument("--num_turns", type=int, default=1,
                          )
  arg_parser.add_argument("--limit_num_problems", type=int, default=None,
                          help="If not none, this is a debug mode where the evaluation happens only on a subset of problems.")
  arg_parser.add_argument("--verbose", type=int, default=0,
                          )
  arg_parser.add_argument("--language", type=str, default="all")
  arg_parser.add_argument("--temp", type=float, default=0.2)
  args = arg_parser.parse_args()
  return args



def construct_messages(turn_idx,
                       problem,
                       execution_result=None,
                       previous_response=None,
                       messages=None,
                       fewshot_examples=[]):
  if turn_idx == 0:
    prompt = problem["prompt"]
  else:
    prompt = "Below is the error message\n-----------------------\n" \
             + execution_result \
             + f"\n-----------------------\nNow, please solve the following problem again\n\n" \
             + problem["prompt"]

  if messages is None:
    assert turn_idx == 0
    # first turn: construct messages
    # (1) high level instruction
    messages = [
      {"role": "system",
       "content": "You are an expert coder in all programming languages. Please continue writing code based on each function signature without repeat the function signature. If you write any explanations in English sentences, please wrap them in comments in the correct format according to that language."},
    ]
    # (2) examples
    for fs_example in fewshot_examples:
      messages.append({"role": "user", "content": fs_example["prompt"]})
      messages.append({"role": "assistant", "content": fs_example["completion"]})
    # (3) add the actual prompt
    messages.append({"role": "user", "content": prompt})
  else:
    # second turn: append user and assistant messages
    assert previous_response is not None, "For next turn, we require providing the previous assistant's response"
    messages.append({"role": "assistant", "content": previous_response})
    messages.append({"role": "user", "content": prompt})
  return messages, prompt


def query_code_completion(model_name,
                          messages,
                          verbose=False,
                          kwargs={},
                          retry_count=5):
  # calling API
  print("Initiating the API call")
  request_result = None
  for i in range(retry_count):
    print(f"API Call number {i + 1}")
    try:
      request_result = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        timeout=10,
        **kwargs,
      )
      break
    except Exception as e:
      print(e)
      time.sleep(2)
  if request_result is None:
    response = ""
    print(f"Warning!! -- API call failed after {retry_count} retries")
  else:
    if verbose:
      print(request_result)
    response = request_result["choices"][0]["message"]["content"]

  return response, messages, request_result #code, request_result


def wrapper_pass_at_k(k_list, total, correct):
  """
  Total:   [num attempts problem 1, num attempts problem 2, ...]
  Correct: [num correct problem 1, num correct problem 2, ...]
  """
  total = np.array(total)
  correct = np.array(correct)
  pass_at_k = {
    f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
    for k in k_list
    if (total >= k).all()
  }
  return pass_at_k


def display_messages(messages):
  for message in messages:
    print(f"{message['role']}:\n{message['content']}\n")


def eval_language(dataset="mbxp",
         model_name="gpt-3.5-turbo",
         language="javascript",
         num_turns=0,
         n_workers=1,
         verbose=False,
         answer_full_function=False,
         execute=True,
         num_reps=1,
         limit_num_problems=None,
         k_shot=2,
         temperature=0):
  check_correctness_function = get_execute_function(language)
  data_obj = get_data(dataset=dataset, language=language)
  fewshot_examples = get_examples(dataset="mbxp", # other datasets have similar few shot examples as mbxp
                                          language=language,
                                          num_examples=k_shot)
  total, correct = {}, {}
  for idx, task_id in enumerate(data_obj):
    if limit_num_problems and idx == limit_num_problems:
      break
    problem = data_obj[task_id]
    passes = []
    for rep in range(num_reps):
      messages, previous_response, execution_result = None, None, None

      for turn_idx in range(num_turns):
        messages, prompt = construct_messages(turn_idx,
                                      problem,
                                      execution_result,
                                      previous_response=previous_response,
                                      messages=messages,
                                      fewshot_examples=fewshot_examples)
        if verbose > 1:
          display_messages(messages)

        response, messages, _ = query_code_completion(
          model_name=model_name,
          messages=messages,
          verbose=False,
          kwargs={"temperature": temperature},
        )

        code = response
        print("\n\n\n\n\n\n")
        print(f"code from query (answer full function ={answer_full_function})")
        print("################################################################################")
        print(prompt)
        print("-----------------[ completion ]------------------")
        print(code)
        print("################################################################################")
        kwargs = {"problem": problem,
                  "completion": code,
                  "timeout": 30.0,
                  "completion_id": 0,
                  }
        if execute:
          with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future = executor.submit(check_correctness_function, **kwargs)
            result = future.result()
            if verbose:
              print(f"Task idx = {task_id} | rep {rep} | passed {result['passed']}")
              print("################################################################################")
            passed = result["passed"]

            if passed:
              break # no need to keep asking in multi turn
            else:
              previous_response = code
              execution_result = result["result"]
              if verbose > 2:
                print("*************** execution result ****************")
                print(execution_result)
                print("**************************************************")
      passes.append(passed)
    if execute:
      assert len(passes) == num_reps, f"Number of passes does not match number of attempts -- {len(passes)} != {num_reps} -- {passes}"
      total[task_id] = num_reps
      correct[task_id] = np.sum(passes)

  # after loop through everything, calculate scores
  if execute:
    # prepare total and correct list
    total_list, correct_list = [], []
    for task_id in total:
      total_list.append(total[task_id])
      correct_list.append(correct[task_id])
    pass_at_k = wrapper_pass_at_k(k_list=[1, num_reps], total=total_list, correct=correct_list)
    print(pass_at_k)


def eval_all_langs():
  args = get_args()
  for language in get_supported_langs(args.dataset):
    if args.language != "all" and language not in args.language:
      continue
    eval_language(dataset=args.dataset,
         model_name=args.model_name,
         language=language,
         verbose=args.verbose, # 2 for most verbose
         limit_num_problems=args.limit_num_problems,
         num_turns=args.num_turns,
         temperature=args.temp,
         k_shot=args.num_shots,
         )

if __name__ == "__main__":
  eval_all_langs()

