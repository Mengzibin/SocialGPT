'''A main script to run attack for LLMs.'''
import time
import importlib
import torch.multiprocessing as mp
from absl import app
import sys
import os
script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory,'llm_prompt'))
from ml_collections import config_flags
from utils.util import get_targets_and_stories,get_targets_and_questions_of_test
from base.prompt_manager import get_workers
_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'llm_prompt.{params.attack}')

    prompts,question_list,answer_list,relation_list = get_targets_and_stories(params)
    question_list_test,answer_list_test,relationship_list_test = get_targets_and_questions_of_test(params)
    question_list_test,answer_list_test,relationship_list_test = question_list_test[:2],answer_list_test[:2],relationship_list_test[:2]

    workers, test_workers = get_workers(params)
    managers = {
        "AP": attack_lib.StoryPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptStory,
    }
    control_init = {
        'system_prompt':prompts['system_prompt'][0],
        'prelude':prompts['prelude'][0],
        'relationship':prompts['relationship'][0],
        'example':prompts['example'][0]
    }
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptStory(
            question_list,
            answer_list,
            relation_list,
            workers,
            control_init,
            prompts,
            progressive_models=params.progressive_models,
            progressive_questions=params.progressive_goals,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_questions=question_list_test,
            test_targets=answer_list_test,
            test_relationships=relationship_list_test,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptStory(
            question_list,
            answer_list,
            relation_list,
            control_init,
            workers,
            prompts,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_questions=question_list_test,
            test_targets=answer_list_test,
            test_relationships=relationship_list_test,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)