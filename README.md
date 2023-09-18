## Code Generation Evaluation
This repository contains a script for evaluating the code generation abilities of GPT models using APIs. The script takes several arguments that allow you to customize the evaluation for your needs. It also allows for multi-turn code evaluation where the model can see the execution error message.

### Usage
To use the script, follow these steps:

* Clone the repository to your local machine.

* Install the required dependencies by running `pip install -r requirements.txt`.

* Open a terminal and navigate to the repository directory.

* mxeval repo for execution in different programming languages. Currently installing directly from source is required. We will support pip installation soon.

```
# Go to a directory where you want this repo installed
git clone https://github.com/amazon-science/mxeval.git
cd mxeval
pip install -e .
```


* Run the script followed by the desired arguments. For example, to evaluate the gpt-3.5-turbo model on the multi-humaneval dataset with num_shots=1 and num_turns=1, enter the following command:


```
python src/eval_gpt_mxeval.py --model_name gpt-3.5-turbo --dataset multi-humaneval --num_shots 1 --num_turns 1
```

* Run Python HumanEval + multi-turn execution (num turns > 1 enables the execution feedback automatically)
```
python src/eval_gpt_mxeval.py --model_name gpt-3.5-turbo --dataset multi-humaneval --num_shots 1 --num_turns 3 --language python --verbose 1
```


### Arguments
The script takes the following arguments:

* `--model_name`: The name of the GPT model to evaluate. Defaults to gpt-3.5-turbo. Options are gpt-3.5-turbo or gpt-4.
* `--dataset`: The dataset to evaluate the model on. Defaults to multi-humaneval. Options are multi-humaneval, mbxp, or mathqa-x.
* `--num_shots`: The number of examples to evaluate the model on. Defaults to 1.
* `--num_turns`: The number of turns per example. Defaults to 1.
* `--limit_num_problems`: If not None, the evaluation will be performed on a subset of problems for debugging purposes. Defaults to None.
* `--verbose`: The verbosity level of the output. Defaults to 0.
* `--language`: The programming language to evaluate the model on. Defaults to all.
* `--temp`: The sampling temperature for the GPT model. Defaults to 0.2.


### API Configuration
To use the script with APIs, you'll need to configure the API credentials in the script. By default, the API key is located here.

```
## YOUR API KEY HERE
openai.api_key_path = ".openai_key"
```


### To Dos
- Log the result to .jsonl file for further analysis