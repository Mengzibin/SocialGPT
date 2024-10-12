# GSPO-PISC

## Dataset

### Overview

Download the PISC dataset online, and use `social-story-main-PISC` to extract the corresponding stories. After story generation, store the story parts of the train and test datasets.

### Structure

Combine the corresponding prompts and stories for GSPO-PISC optimization to extract the most effective prompts. Templates for prompts are provided in the paper, and a set of candidate prompts should be prepared.

### Directory Layout

Here's the folder structure containing prompts and stories:

```plaintext
dataset-pisc/
├── definition_of_example/        # Contains examples for model optimization
├── definition_of_relationship/   # Contains relationship definitions
├── prelude/                      # Initial setup scripts
├── system_prompt/                # System prompts for guiding the LLM's responses
├── story_train/                  # Train dataset after story generation
└── story_test/                   # Test dataset after story generation
```

## Code

### Configuration

The file `experiment/configs/transfer_vicuna.py` includes the following key configurations:

| Parameter           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `tokenizer_paths`   | Location of `vicuna-7b-v1.5-16k` folder                      |
| `model_paths`       | Same as above                                                |
| `image_txt`         | Train part's relation file from the PISC dataset             |
| `image_txt_test`    | Test part's relation file from the PISC dataset              |
| `n_steps`           | Number of optimization iterations, set above 500             |
| `sam4story`         | Folder location for generated stories and prompts            |
| `example_dir`       | Folder for storing examples in the prompt                    |
| `relationship_dir`  | Folder for storing relationship definitions in the prompt    |
| `prelude_dir`       | Folder for initial setup scripts in the prompt               |
| `system_prompt_dir` | Folder for system prompts in the prompt                      |
| `story_dir`         | Folder for train stories extracted via `social-story-main-PISC` |
| `story_dir_test`    | Folder for test stories extracted via `social-story-main-PISC` |

### Acquisition

Obtain the `vicuna-7b-v1.5-16k` folder by downloading from the HuggingFace official website.

### Execution Scripts

The file `experiment/launch_scripts/run_story_10.sh` is an example shell script for running prompt optimization.

## Run the Code

### Environment Setup

Create a new virtual environment called GSPO with conda.

### Dependencies

Install all libraries from the `requirements.txt` file.

### Running Scripts

Navigate to the scripts directory and execute the script:

```bash
cd experiments/launch_scripts
bash run_story_10.sh vicuna
```

## Result

### Output

The optimized prompts post GSPO-PISC optimization will be stored in the `experiments` folder.