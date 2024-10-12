import gc
import json
import math
import time
import random
import torch
from copy import deepcopy
from typing import Optional, Any

import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)
from fastchat.model import get_conversation_template

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings_input_ids_attn_mask(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)

class ModelWorker(object):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        ).to(device).eval()
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue()
        self.process = None
    
    @staticmethod
    def run(model, tasks, results):
        while True:
            task = tasks.get()
            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

    def start(self):
        self.process = mp.Process(
            target=ModelWorker.run,
            args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        print(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self
    
    def stop(self):
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self

def get_workers(params,eval=False):
    tokenizers = []
    for i in range(len(params.tokenizer_paths)):
        tokenizer = AutoTokenizer.from_pretrained(
            params.tokenizer_paths[i],
            trust_remote_code=True,
            **params.tokenizer_kwargs[i]
        )
        if 'oasst-sft-6-llama-30b' in params.tokenizer_paths[i]:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        if 'guanaco' in params.tokenizer_paths[i]:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        if 'llama-2' in params.tokenizer_paths[i]:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'falcon' in params.tokenizer_paths[i]:
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append(tokenizer)
    
    print(f"Loaded {len(tokenizers)} tokenizers")

    raw_conv_templates = [
        get_conversation_template(template)
        for template in params.conversation_templates
    ]

    conv_templates = []
    for conv in raw_conv_templates:
        if conv.name == 'zero_shot':
            conv.roles = tuple(['### ' + r for r in conv.roles])
            conv.sep = '\n'
        elif conv.name == 'llama-2':
            conv.sep2 = conv.sep2.strip()
        conv_templates.append(conv)

    print(f'Loaded {len(conv_templates)} conversation templates')
    workers = [
        ModelWorker(
            params.model_paths[i],
            params.model_kwargs[i],
            tokenizers[i],
            conv_templates[i],
            params.devices[i]
        )
        for i in range(len(params.model_paths))
    ]

    if not eval:
        for worker in workers:
            worker.start()
    num_train_models = getattr(params, 'num_train_models', len(workers))
    print('Loaded {} train models'.format(num_train_models))
    print('Loaded {} test models'.format(len(workers) - num_train_models))

    return workers[:num_train_models], workers[num_train_models:]

class StoryPrompt(object):
    """
    A class used to generate an story prompt
    """
    def __init__(self, 
        questions,
        target,
        relationship_answer,
        control_init,
        controls,
        tokenizer,
        conv_template,
        test_prefixes = ["I'm sorry","I apologize","As an","I'm just","I'm an","I cannot","Hello!"],
        *args, **kwargs
    ):
        self.controls = controls
        self.story = questions['story']
        self.name1 = questions['R1']
        self.name2 = questions['R2']
        self.example = control_init['example']
        self.relationship = control_init['relationship']
        self.prelude = control_init['prelude']
        self.system_prompt = control_init['system_prompt']
        self.target = target
        self.relationship_answer = relationship_answer
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.test_prefixes = test_prefixes

        self.conv_template.messages = []

        self.test_new_toks = len(self.tokenizer(self.target).input_ids) + 2
        for prefix in self.test_prefixes:
            self.test_new_toks = max(self.test_new_toks, len(self.tokenizer(prefix).input_ids))
        self._get_voiceover()
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
        
    def _update_ids(self):
        self.conv_template.system_message = self.system_prompt
        self.conv_template.append_message(self.conv_template.roles[0], f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}{self.example}{self.after_example}{self.story}{self.after_story}{self.name1}{self.after_name1}{self.name2}{self.after_name2}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")

        prompt = self.conv_template.get_prompt()
        encoding = self.tokenizer(prompt)

        toks = encoding.input_ids
        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []
            self.conv_template.system_message = ''

            self.conv_template.system_message = self.system_prompt
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            #self._system_prompt_slice = slice(None,len(toks))
            self._system_prompt_slice = slice(1,len(toks)-1)

            self.conv_template.append_message(self.conv_template.roles[0],None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(self._system_prompt_slice.stop,len(toks))

            self.conv_template.update_last_message(f"{self.prelude}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._prelude_slice = slice(self._user_role_slice.stop,max(self._user_role_slice.stop,len(toks)))

            self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._after_prelude_slice = slice(self._prelude_slice.stop,len(toks))

            self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._relationship_slice = slice(self._after_prelude_slice.stop,len(toks))

            self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._after_relationship_slice = slice(self._relationship_slice.stop,len(toks))

            self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}{self.example}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._example_slice = slice(self._after_relationship_slice.stop,len(toks))

            self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}{self.example}{self.after_example}{self.story}{self.after_story}{self.name1}{self.after_name1}{self.name2}{self.after_name2}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._question_slice = slice(self._example_slice.stop,len(toks))

            self.conv_template.append_message(self.conv_template.roles[1],None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._question_slice.stop,len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop,len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1,len(toks)-3)

        else:

            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True
            if python_tokenizer:

                self.conv_template.messages = []
                self.conv_template.system_message = ''

                self.conv_template.system_message = self.system_prompt
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                #self._system_prompt_slice = slice(None,len(toks))
                self._system_prompt_slice = slice(1,len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[0],None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(self._system_prompt_slice.stop,len(toks))

                self.conv_template.update_last_message(f"{self.prelude}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._prelude_slice = slice(self._user_role_slice.stop,max(self._user_role_slice.stop,len(toks)-1))

                self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._after_prelude_slice = slice(self._prelude_slice.stop,len(toks)-1)

                self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._relationship_slice = slice(self._after_prelude_slice.stop,len(toks)-1)

                self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._after_relationship_slice = slice(self._relationship_slice.stop,len(toks)-1)
                
                self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}{self.example}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._example_slice = slice(self._after_relationship_slice.stop,len(toks)-1)

                self.conv_template.update_last_message(f"{self.prelude}{self.after_prelude}{self.relationship}{self.after_relationship}{self.example}{self.after_example}{self.story}{self.after_story}{self.name1}{self.after_name1}{self.name2}{self.after_name2}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._question_slice = slice(self._example_slice.stop,len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1],None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._question_slice.stop,len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop,len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1,len(toks)-2)

            else:
                self._system_slice = slice(
                    None,
                    encoding.char_to_token(len(self.conv_template.system_message))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._prelude_slice = slice(
                    encoding.char_to_token(prompt.find(self.prelude)),
                    encoding.char_to_token(prompt.find(self.prelude) + len(self.prelude))
                )
                self._after_prelude_slice = slice(
                    encoding.char_to_token(prompt.find(self.after_prelude)),
                    encoding.char_to_token(prompt.find(self.after_prelude) + len(self.after_prelude))
                )
                self._relationship_slice = slice(
                    encoding.char_to_token(prompt.find(self.relationship)),
                    encoding.char_to_token(prompt.find(self.relationship) + len(self.relationship))
                )
                self._after_relationship_slice = slice(
                    encoding.char_to_token(prompt.find(self.after_relationship)),
                    encoding.char_to_token(prompt.find(self.after_relationship) + len(self.after_relationship))
                )
                self._example_slice = slice(
                    encoding.char_to_token(prompt.find(self.example)),
                    encoding.char_to_token(prompt.find(self.example) + len(self.example))
                )
                self._question_slice = slice(
                    encoding.char_to_token(prompt.find(self.after_example+self.story+self.after_story+self.name1+self.after_name1+self.name2+self.after_name2)),
                    encoding.char_to_token(prompt.find(self.after_example+self.story+self.after_story+self.name1+self.after_name1+self.name2+self.after_name2) + len(self.after_example+self.story+self.after_story+self.name1+self.after_name1+self.name2+self.after_name2))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )
        self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
        self.conv_template.messages = []
    
    def _input_id_add_mele_toks(self,num=400):
        pad_tok = 0
        while pad_tok in self.input_ids:
            pad_tok += 1
        self._pad_tok = pad_tok
        pad_tensor = torch.full([num],pad_tok)
        self.input_ids = torch.concatenate([pad_tensor,self.input_ids],axis=0)

        self._pad_tensor_slice = slice(None, num)
        self._system_prompt_slice = slice(self._system_prompt_slice.start+num,self._system_prompt_slice.stop+num)
        self._prelude_slice = slice(self._prelude_slice.start+num,self._prelude_slice.stop+num)
        self._after_prelude_slice = slice(self._after_prelude_slice.start+num,self._after_prelude_slice.stop+num)
        self._relationship_slice = slice(self._relationship_slice.start+num,self._relationship_slice.stop+num)
        self._after_relationship_slice = slice(self._after_relationship_slice.start+num,self._after_relationship_slice.stop+num)
        self._example_slice = slice(self._example_slice.start+num,self._example_slice.stop+num)
        self._question_slice = slice(self._question_slice.start+num,self._question_slice.stop+num)
        self._assistant_role_slice = slice(self._assistant_role_slice.start+num,self._assistant_role_slice.stop+num)
        self._target_slice = slice(self._target_slice.start+num,self._target_slice.stop+num)
        self._loss_slice = slice(self._loss_slice.start+num,self._loss_slice.stop+num)

    @property
    def pad_tok(self):
        return self._pad_tok
    
    @pad_tok.setter
    def pad_tok(self,pad_tok):
        self.input_ids = torch.where(self.input_ids == self._pad_tok,pad_tok,self.input_ids)
        self._pad_tok = pad_tok

    @torch.no_grad()
    def generate(self,model,gen_config = None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 300
        
        if gen_config.max_new_tokens > 400:
            print('WARNING: max_new_tokens > 400 may cause testing to slow down.')
        
        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = (torch.ne(input_ids, self._pad_tok)).type(input_ids.dtype).to(model.device)
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id)[0]
        return output_ids[self._assistant_role_slice.stop:]

    def generate_str(self,model,gen_config=None):
        return self.tokenizer.decode(self.generate(model,gen_config))

    def test(self,model,gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = self.test_new_toks
        
        gen_str = self.generate_str(model,gen_config).strip()
        jailbroken = not any([prefix in gen_str for prefix in self.test_prefixes]) and self.relationship_answer in gen_str
        em = self.target in gen_str
        return jailbroken, int(em)
    
    @torch.no_grad()
    def test_loss(self, model):
        logits, ids = self.logits(model, return_ids=True)
        return self.target_loss(logits, ids).mean().item()
    
    def grad(self,model):
        return NotImplementedError("Gradient function not yet implemented")

    def _get_voiceover(self):
        self.after_prelude = '\n\n'
        self.after_relationship = '\n\n****************************************************************\nExample:\n'
        self.after_example = '\n****************************************************************\n\n[1. image description]:\n'
        self.after_story = '\n\n[2. Question]: \nWhat are the most likely social relationships between '
        self.after_name1 = ' and '
        self.after_name2 = '? Choose only one from {<father-child>, <mother-child>, <grandpa-grandchild>, <grandma-grandchild>, <friends>, <siblings>, <classmates>, <lovers/spouses>, <presenter-audience>, <teacher-student>, <trainer-trainee>, <leader-subordinate>, <band members>, <dance team members>, <sport team members>, <colleagues>}..\n\n[3. Answer]:'

    @torch.no_grad()
    def logits(self,model,test_controls=None,return_ids=False):
        if test_controls is None:
            test_controls = self.control_toks
        if isinstance(test_controls['prelude'], torch.Tensor):
            if len(test_controls['prelude'].shape) == 1:
                test_controls = self.control_toks_unsqueeze(test_controls)
            test_ids = self.control_to_model_device(test_controls,model)
        elif not isinstance(test_controls['prelude'],list):
            test_controls = self.control_to_list(test_controls)
        elif isinstance(test_controls['prelude'][0],str):
            test_id_controls = {}
            for key in test_controls.keys():
                test_id_controls[key] = [
                    torch.tensor(self.tokenizer(text,add_special_tokens=False).input_ids)
                    for text in test_controls[key]
                ]
            pad_tok = 0
            temp_ids = []
            for i in range(len(test_id_controls['prelude'])):
                temp_ids.append(
                    torch.cat(
                        [
                            test_id_controls['system_prompt'][i],
                            self.input_ids[self._system_prompt_slice.stop:self._prelude_slice.start],
                            test_id_controls['prelude'][i],
                            self.input_ids[self._prelude_slice.stop:self._relationship_slice.start],
                            test_id_controls['relationship'][i],
                            self.input_ids[self._relationship_slice.stop:self._example_slice.start],
                            test_id_controls['example'][i],
                            self.input_ids[self._example_slice.stop:]
                        ],
                        dim=0
                    )
                )
            maxlen = max([temp_ids[i].shape[0] for i in range(len(temp_ids))])
            maxlen = max([maxlen,self.input_ids.shape[0]])
            ids = []
            for i in range(len(temp_ids)):
                num_specific_elements = maxlen - temp_ids[i].shape[0]
                specific_elements = torch.full((num_specific_elements,),pad_tok)
                expanded_vector = torch.cat([specific_elements,temp_ids[i]])
                ids.append(list(expanded_vector))
            ids = torch.tensor(ids).to(model.device)
            attn_mask = (torch.ne(ids, pad_tok)).type(torch.int64)
            del test_id_controls,temp_ids; gc.collect()
        else:
            raise ValueError(f"test_controls must be a list of strings or a tensor of token ids, got {type(test_controls['prelude'])}")

        if not isinstance(test_controls['prelude'][0],str):
            for key in test_ids.keys():
                key_start = int(eval('self._'+key+'_slice.start')) if eval('self._'+key+'_slice.start') is not None else 0
                key_stop = int(eval('self._'+key+'_slice.stop'))
                if not(test_ids[key][0].shape[0] == (key_stop - key_start)):
                    raise ValueError((
                        f"test_controls "+key+" must have shape "
                        f"(n, {key_stop - key_start})" 
                        f"got {test_ids[key].shape} {test_ids[key]}"
                    ))
            ids = self.input_ids.unsqueeze(0).repeat(test_ids['prelude'].shape[0], 1).to(model.device)
            if test_ids['prelude'][0].shape[0] == int(eval('self._prelude_slice.stop')) - int(eval('self._prelude_slice.start')):
                for key in test_ids.keys():
                    key_start = int(eval('self._'+key+'_slice.start')) if eval('self._'+key+'_slice.start') is not None else 0
                    key_stop = int(eval('self._'+key+'_slice.stop'))
                    locs = torch.arange(key_start,key_stop).repeat(test_ids[key].shape[0], 1).to(model.device)
                    ids = torch.scatter(
                        ids,
                        1,
                        locs,
                        test_ids[key]
                    )
            if test_ids['prelude'][0].shape[0] == int(eval('self._prelude_slice.stop')) - int(eval('self._prelude_slice.start')):
                attn_mask = (torch.ne(ids, self._pad_tok)).type(ids.dtype)
            del locs,test_ids; gc.collect()
        if return_ids:
            torch.cuda.empty_cache()
            return model(input_ids=ids, attention_mask=attn_mask).logits, ids
        else:
            logits = model(input_ids=ids, attention_mask=attn_mask).logits
            del ids ; gc.collect()
            return logits

    def _get_input_ids_without_pad_tok(self):
        attn_mask = (torch.ne(self.input_ids, self._pad_tok)).type(self.input_ids.dtype)
        attn_mask_nonzero_position = torch.nonzero(attn_mask==1,as_tuple=True)
        input_ids_without_pad_tok = self.input_ids[attn_mask_nonzero_position]
        different_value = self.input_ids.shape[0] - input_ids_without_pad_tok.shape[0]
        system_prompt_slice = slice(self._system_prompt_slice.start-different_value,self._system_prompt_slice.stop-different_value)
        prelude_slice = slice(self._prelude_slice.start-different_value,self._prelude_slice.stop-different_value)
        relationship_slice = slice(self._relationship_slice.start-different_value,self._relationship_slice.stop-different_value)
        example_slice = slice(self._example_slice.start-different_value,self._example_slice.stop-different_value)
        target_slice = slice(self._target_slice.start-different_value,self._target_slice.stop-different_value)
        loss_slice = slice(self._loss_slice.start-different_value,self._loss_slice.stop-different_value)
        return input_ids_without_pad_tok,system_prompt_slice,prelude_slice,relationship_slice,example_slice,target_slice,loss_slice

    def target_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction="none")
        loss_slice = slice(self._target_slice.start-1,self._target_slice.stop-1)
        loss = crit(logits[:,loss_slice,:].transpose(1,2),ids[:,self._target_slice])
        return loss
    
    def control_to_model_device(self,control,model):
        control['prelude'] = control['prelude'].to(model.device)
        control['relationship'] = control['relationship'].to(model.device)
        control['example'] = control['example'].to(model.device)
        control['system_prompt'] = control['system_prompt'].to(model.device)
        return control
        
    def control_to_list(self,control):
        control['prelude'] = [control['prelude']]
        control['relationship'] = [control['relationship']]
        control['example'] = [control['example']]
        control['system_prompt'] = [control['system_prompt']]
        return control
    
    def control_toks_unsqueeze(self,control):
        control['prelude'] = control['prelude'].unsqueeze(0)
        control['relationship'] = control['relationship'].unsqueeze(0)
        control['example'] = control['example'].unsqueeze(0)
        control['system_prompt'] = control['system_prompt'].unsqueeze(0)
        return control

    @property
    def control_str(self):
        control = {}
        control['prelude'] = self.tokenizer.decode(self.input_ids[self._prelude_slice]).strip()
        control['relationship'] = self.tokenizer.decode(self.input_ids[self._relationship_slice]).strip()
        control['example'] = self.tokenizer.decode(self.input_ids[self._example_slice]).strip()
        control['system_prompt'] = self.tokenizer.decode(self.input_ids[self._system_prompt_slice]).strip()
        return control
    
    @property
    def control_toks(self):
        control = {}
        control['prelude'] = self.input_ids[self._prelude_slice]
        control['relationship'] = self.input_ids[self._relationship_slice]
        control['example'] = self.input_ids[self._example_slice]
        control['system_prompt'] = self.input_ids[self._system_prompt_slice]
        return control
    
    @control_str.setter
    def control_str(self,control):
        self.prelude = control['prelude']
        self.relationship = control['relationship']
        self.example = control['example']
        self.system_prompt = control['system_prompt']
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @control_toks.setter
    def control_toks(self,control):
        self.prelude = self.tokenizer.decode(control['prelude'])
        self.relationship = self.tokenizer.decode(control['relationship'])
        self.example = self.tokenizer.decode(control['example'])
        self.system_prompt = self.tokenizer.decode(control['system_prompt'])
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

    @property
    def system_prompt_str(self):
        return self.tokenizer.decode(self.input_ids[self._system_prompt_slice]).strip()

    @system_prompt_str.setter
    def system_prompt_str(self,system_prompt):
        self.system_prompt = system_prompt
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @property
    def system_prompt_toks(self):
        return self.input_ids[self._system_prompt_slice]
    
    @system_prompt_toks.setter
    def system_prompt_toks(self,system_prompt_toks):
        self.system_prompt = self.tokenizer.decode(system_prompt_toks)
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

    @property
    def eval_str(self):
        return self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop]).replace('<s>','').replace('</s>','')
    
    @property
    def input_str(self):
        return self.tokenizer.decode(self.input_ids)
    
    @property
    def input_toks(self):
        return self.input_ids
    
    @property
    def prompt(self):
        return self.tokenizer.decoder(self.input_ids[self._prelude_slice.start:self._question_slice.stop])
    
    @property
    def prelude_toks(self):
        return self.input_ids[self._prelude_slice]
    
    @prelude_toks.setter
    def prelude_toks(self,prelude_toks):
        self.prelude = self.tokenizer.decode(prelude_toks)
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

    @property
    def prelude_str(self):
        return self.tokenizer.decode(self.input_ids[self._prelude_slice]).strip()
    
    @prelude_str.setter
    def prelude_str(self,prelude):
        self.prelude = prelude
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @property
    def assistant_str(self):
        return self.tokenizer.decode(self.input_ids[self._assistant_role_slice]).strip()
    
    @property
    def assistant_toks(self):
        return self.input_ids[self._assistant_role_slice]
    
    @property
    def relationship_str(self):
        return self.tokenizer.decode(self.input_ids[self._relationship_slice]).strip()
    
    @property
    def relationship_toks(self):
        return self.input_ids[self._relationship_slice]
    
    @relationship_str.setter
    def relationship_str(self,relationship):
        self.relationship = relationship
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @relationship_toks.setter
    def relationship_toks(self,relationship_toks):
        self.relationship = self.tokenizer.decode(relationship_toks)
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

    @property
    def example_str(self):
        return self.tokenizer.decode(self.input_ids[self._example_slice]).strip()
    
    @property
    def example_toks(self):
        return self.input_ids[self._example_slice]

    @example_str.setter
    def example_str(self,example):
        self.example = example
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @example_toks.setter
    def example_toks(self,example_toks):
        self.example = self.tokenizer.decode(example_toks)
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @property
    def after_prelude_str(self):
        return self.tokenizer.decode(self.input_ids[self._after_prelude_slice]).strip()
    
    @property
    def after_prelude_toks(self):
        return self.input_ids[self._after_prelude_slice]

    @after_prelude_str.setter
    def after_prelude_str(self,after_prelude):
        self.after_prelude = after_prelude
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

    @after_prelude_toks.setter
    def after_prelude_toks(self,after_prelude_toks):
        self.after_prelude = self.tokenizer.decode(after_prelude_toks)
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @property
    def after_relationship_str(self):
        return self.tokenizer.decode(self.input_ids[self._after_relationship_slice]).strip()
    
    @property
    def after_relationship_toks(self):
        return self.input_ids[self._after_relationship_slice]

    @after_relationship_str.setter
    def after_relationship_str(self,after_relationship):
        self.after_relationship = after_relationship
        self._update_ids()
        self._input_id_add_mele_toks(num=400)
    
    @after_relationship_toks.setter
    def after_relationship_toks(self,after_relationship_toks):
        self.after_relationship = self.tokenizer.decode(after_relationship_toks)
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

    @property
    def question_str(self):
        return self.tokenizer.decode(self.input_ids[self._question_slice]).strip()
    
    @property
    def question_toks(self):
        return self.input_ids[self._question_slice]

    @question_str.setter
    def question_str(self,question):
        self.story = question['story']
        self.name1 = question['R1']
        self.name2 = question['R2']
        self._update_ids()
        self._input_id_add_mele_toks(num=400)

class PromptManager(object):
    def __init__(self,
        questions,
        targets,
        relationships,
        control_init,
        controls,
        tokenizer,
        conv_template,
        test_prefixes = ["I'm sorry","I apologize","As an","I'm just","I'm an","I cannot","Hello!"],
        managers=None,
        *args, **kwargs
    ):
        if len(questions) != len(targets):
            raise ValueError("Length of questions and targets must match")
        if len(questions) == 0:
            raise ValueError("Must provide at least one question, target pair")
        self.tokenizer = tokenizer
        self.controls = controls
        self._prompts = [
            managers['AP'](
                question,
                target,
                relationship,
                control_init,
                controls,
                tokenizer,
                conv_template,
                test_prefixes
            )
            for question, target, relationship in zip(questions, targets, relationships)
        ]
        self._nonascii_toks = get_nonascii_toks(tokenizer, device='cpu')
    
    def generate(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 300

        return [prompt.generate(model, gen_config) for prompt in self._prompts]
    
    def generate_str(self, model, gen_config=None):
        return [
            self.tokenizer.decode(output_toks) 
            for output_toks in self.generate(model, gen_config)
        ]
    
    def test(self, model, gen_config=None):
        return [prompt.test(model, gen_config) for prompt in self._prompts]

    def test_loss(self, model):
        return [prompt.test_loss(model) for prompt in self._prompts]
    
    def grad(self, model):
        grads = [prompt.grad(model) for prompt in self._prompts]
        grads_return = {}
        for key in grads[0].keys():
            grads_return[key] = sum([grads[i][key] for i in range(len(grads))])
        return grads_return
    
    def logits(self, model, test_controls=None, return_ids=False):
        vals = [prompt.logits(model, test_controls, return_ids) for prompt in self._prompts]
        if return_ids:
            return [val[0] for val in vals], [val[1] for val in vals]
        else:
            return vals
    
    def target_loss(self, logits, ids):
        for prompt, logit, id in zip(self._prompts, logits, ids):
            print(prompt.target_loss(logit, id))
        return torch.cat(
            [
                prompt.target_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)
    
    def sample_control(self, *args, **kwargs):

        raise NotImplementedError("Sampling control tokens not yet implemented")

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, i):
        return self._prompts[i]

    def __iter__(self):
        return iter(self._prompts)
    
    @property
    def control_str(self):
        return self._prompts[0].control_str
    
    @property
    def control_toks(self):
        return self._prompts[0].control_toks

    @control_str.setter
    def control_str(self, control):
        for prompt in self._prompts:
            prompt.control_str = control
    
    @control_toks.setter
    def control_toks(self, control_toks):
        for prompt in self._prompts:
            prompt.control_toks = control_toks

    @property
    def disallowed_toks(self):
        return self._nonascii_toks

class MultiPromptStory(object):
    def __init__(self,
        questions,
        targets,
        relationships,
        control_init,
        controls,
        workers,
        test_prefixes = ["I'm sorry","I apologize","As an","I'm just","I'm an","I cannot","Hello!"],
        logfile=None,
        managers=None,
        test_questions=[],
        test_targets=[],
        test_relationships=[],
        test_workers=[],
        *args, **kwargs
    ):
        self.controls = controls
        self.questions = questions
        self.relationships = relationships
        self.control_init = control_init
        self.targets = targets
        self.workers = workers
        self.test_questions = test_questions
        self.test_targets = test_targets
        self.test_relationships = test_relationships
        self.test_workers = test_workers
        self.test_prefixes = test_prefixes
        self.models = [worker.model for worker in workers]
        self.logfile = logfile
        self.prompts = [
            managers['PM'](
                questions,
                targets,
                relationships,
                control_init,
                controls,
                worker.tokenizer,
                worker.conv_template,
                test_prefixes,
                managers
            )
            for worker in workers
        ]
        self.managers = managers

    @property
    def control_str(self):
        return self.prompts[0].control_str
    
    @control_str.setter
    def control_str(self, control):
        for prompts in self.prompts:
            prompts.control_str = control
    
    @property
    def control_toks(self):
        return [prompts.control_toks for prompts in self.prompts]
    
    @control_toks.setter
    def control_toks(self, control):
        if len(control) != len(self.prompts):
            raise ValueError("Must provide control tokens for each tokenizer")
        for i in range(len(control)):
            self.prompts[i].control_toks = control[i]
    
    def get_filtered_cands(self, worker_index, control_cand, filter_cand=True, curr_control=None):
        new_cands = {}
        for key in control_cand.keys():
            cands, count = [], 0
            for i in range(len(control_cand[key])):
                decoded_str = control_cand[key][i]
                if filter_cand:
                    if decoded_str != curr_control[key]:
                        cands.append(decoded_str)
                    else:
                        count += 1
                else:
                    cands.append(decoded_str)
            if filter_cand:
                cands = cands + [cands[-1]] * (len(control_cand[key]) - len(cands))
                print(f"Warning: {round(count / len(control_cand[key]), 2)} {key} control candidates were not valid")
            new_cands[key] = cands
        return new_cands

    def step(self,*args,**kwargs):
        return NotImplementedError('Story step function not yet implemented')
    
    def run(self,
        n_steps=100,
        batch_size=1024,
        topk=256,
        temp=1,
        allow_non_ascii=True,
        target_weight=None,
        anneal=True,
        anneal_from=0,
        prev_loss=np.infty,
        stop_on_success=True,
        test_steps=50,
        log_first=False,
        filter_cand=True,
        verbose=True
    ):
        def P(e,e_prime,k):
            T = max(1 - float(k+1)/(n_steps+anneal_from), 1.e-7)
            return True if e_prime < e else math.exp(-(e_prime-e)/T) >= random.random()

        if target_weight is None:
            target_weight_fn = lambda _: 1
        elif isinstance(target_weight, (int,float)):
            target_weight_fn = lambda i: target_weight
        steps = 0
        loss = best_loss = 1e6
        best_control = self.control_str
        runtime = 0.

        if self.logfile is not None and log_first:
            model_tests = self.test_all()

        for i in range(n_steps):
            if stop_on_success:
                model_tests_jb,model_tests_mb,_ = self.test(self.workers,self.prompts)
                if all(all(tests for tests in model_test) for model_test in model_tests_jb):
                    break
            steps += 1
            start = time.time()
            torch.cuda.empty_cache()
            control,loss = self.step(
                batch_size=batch_size,
                topk=topk, 
                temp=temp, 
                allow_non_ascii=allow_non_ascii, 
                target_weight=target_weight_fn(i), 
                filter_cand=filter_cand,
                verbose=verbose
            )
            runtime = time.time() - start
            keep_control = True if not anneal else P(prev_loss, loss, i+anneal_from)
            if keep_control:
                self.control_str = control
            prev_loss = loss

            if loss < best_loss:
                best_loss = loss
                best_control = control
            print('Current Loss:',loss,'Best Loss:',best_loss)
        return self.control_str, loss, steps

    def test(self,workers,prompts,include_loss=False):
        
        for j, worker in enumerate(workers):
            worker(prompts[j],"test",worker.model)
        model_tests = np.array([worker.results.get() for worker in workers])
        model_tests_jb = model_tests[...,0].tolist()
        model_tests_mb = model_tests[...,1].tolist()
        model_tests_loss = []
        if include_loss:
            for j, worker in enumerate(workers):
                worker(prompts[j], "test_loss", worker.model)
            model_tests_loss = [worker.results.get() for worker in workers]

        return model_tests_jb, model_tests_mb, model_tests_loss

    def test_all(self):
        all_workers = self.workers + self.test_workers
        all_prompts = [
            self.managers['PM'](
                self.questions + self.test_questions,
                self.targets + self.test_targets,
                self.relationships + self.test_relationships,
                self.control_str,
                self.controls,
                worker.tokenizer,
                worker.conv_template,
                self.test_prefixes,
                self.managers
            )
            for worker in all_workers
        ]
        return self.test(all_workers,all_prompts,include_loss=True)

    def parse_results(self, results):
        x = len(self.workers)
        i = len(self.questions)
        id_id = results[:x, :i].sum()
        id_od = results[:x, i:].sum()
        od_id = results[x:, :i].sum()
        od_od = results[x:, i:].sum()
        return id_id, id_od, od_id, od_od

class IndividualPromptStory(object):
    def __init__(self,
        questions,
        targets,
        relationships,
        control_init,
        workers,
        controls,
        test_prefixes=["I'm sorry","I apologize","As an","I'm just","I'm an","I cannot","Hello!"],
        logfile=None,
        managers=None,
        test_questions=[],
        test_targets=[],
        test_relationships=[],
        test_workers=[],
        *args,
        **kwargs
    ):
        self.questions = questions
        self.controls = controls
        self.control_init = control_init
        self.targets = targets
        self.relationships = relationships
        self.workers = workers
        self.test_questions = test_questions
        self.test_targets = test_targets
        self.test_relationships = test_relationships
        self.test_workers = test_workers
        self.test_prefixes = test_prefixes
        self.logfile = logfile
        self.managers = managers
        self.mpa_kwargs = IndividualPromptStory.filter_mpa_kwargs(**kwargs)

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs
    
    def run(self, 
            n_steps: int = 1000, 
            batch_size: int = 1024, 
            topk: int = 256, 
            temp: float = 1., 
            allow_non_ascii: bool = True, 
            target_weight: Optional[Any] = None, 
            anneal: bool = True, 
            test_steps: int = 50, 
            incr_control: bool = True,
            stop_on_success: bool = True,
            verbose: bool = True,
            filter_cand: bool = True
        ):
        
        stop_inner_on_success = stop_on_success
        for i in range(len(self.questions)):
            print(f"Question {i+1}/{len(self.questions)}")
            attack = self.managers['MPA'](
                self.questions[i:i+1],
                self.targets[i:i+1],
                self.relationships[i:i+1],
                self.control_init,
                self.controls,
                self.workers,
                self.test_prefixes,
                self.logfile,
                self.managers,
                self.test_questions,
                self.test_targets,
                self.test_relationships,
                self.test_workers,
                **self.mpa_kwargs
            )
            attack.run(
                n_steps=n_steps,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                anneal=anneal,
                anneal_from=0,
                prev_loss=np.infty,
                stop_on_success=stop_inner_on_success,
                test_steps=test_steps,
                log_first=True,
                filter_cand=filter_cand,
                verbose=verbose
            )
        return self.control,n_steps

class ProgressiveMultiPromptStory(object):
    def __init__(self,
        questions,
        targets,
        relationships,
        workers,
        control_init,
        controls,
        progressive_questions=True,
        progressive_models=True,
        test_prefixes=["I'm sorry","I apologize","As an","I'm just","I'm an","I cannot","Hello!"],
        logfile=None,
        managers=None,
        test_questions=[],
        test_targets=[],
        test_relationships=[],
        test_workers=[],
        *args, **kwargs
    ):
        self.questions = questions
        self.targets = targets
        self.relationships = relationships
        self.workers = workers
        self.control_init = control_init
        self.controls = controls
        self.test_questions = test_questions
        self.test_targets = test_targets
        self.test_relationships = test_relationships
        self.test_workers = test_workers
        self.progressive_questions = progressive_questions
        self.progressive_models = progressive_models
        self.test_prefixes = test_prefixes
        self.logfile = logfile
        self.managers = managers
        self.mpa_kwargs = ProgressiveMultiPromptStory.filter_mpa_kwargs(**kwargs)

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    def run(self, 
            n_steps: int = 1000, 
            batch_size: int = 1024, 
            topk: int = 256, 
            temp: float = 1., 
            allow_non_ascii: bool = False, 
            target_weight = None, 
            anneal: bool = True, 
            test_steps: int = 50, 
            incr_control: bool = True, 
            stop_on_success: bool = True, 
            verbose: bool = True, 
            filter_cand: bool = True, 
        ):
        
        num_questions = 2 if self.progressive_questions else len(self.questions)
        num_workers = 1 if self.progressive_models else len(self.workers)
        step = 0
        stop_inner_on_success = self.progressive_questions
        loss = best_loss = np.infty

        def get_random_idx(relationships_total):
            str_to_label = {'father-child':0,'mother-child':1,'grandpa-grandchild':2,'grandma-grandchild':3,
                'friends':4,'siblings':5,'classmates':6,'lovers/spouses':7,
                'presenter-audience':8,'teacher-student':9,'trainer-trainee':10,'leader-subordinate':11,
                'band members':12,'dance team members':13,'sport team members':14,'colleagues':15}
            relationship_idx = {}
            idx = []
            
            for key in str_to_label.keys():
                relationship_idx[key] = []
            for i in range(len(relationships_total)):
                relationship_idx[relationships_total[i]].append(i)
            for key in relationship_idx.keys():
                if len(relationship_idx[key]) > 0:
                    idx.append(random.choice(relationship_idx[key]))
            return idx
        
        self.control = self.control_init

        while step < n_steps:

            step += 1
            questions_total = self.questions
            relationships_total = self.relationships
            targets_total = self.targets
            
            idx = random.sample(list(range(len(relationships_total))),8)
            questions = [questions_total[i] for i in idx]
            targets = [targets_total[i] for i in idx]
            relationships = [relationships_total[i] for i in idx]

            attack = self.managers['MPA'](
                questions,
                targets,
                relationships,
                self.control,
                self.controls,
                self.workers[:num_workers],
                self.test_prefixes,
                self.logfile,
                self.managers,
                self.test_questions,
                self.test_targets,
                self.test_relationships,
                self.test_workers,
                **self.mpa_kwargs
            )

            control,loss,inner_steps = attack.run(
                n_steps=1,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                anneal=anneal,
                anneal_from=step,
                prev_loss=loss,
                stop_on_success=stop_inner_on_success,
                test_steps=test_steps,
                filter_cand=filter_cand,
                verbose=verbose
            )

            self.control = control
            if loss < best_loss:
                best_loss = loss
                best_control = control
            print('Current Loss:',loss,'Best Loss:',best_loss)

            if n_steps % step == 0:
                model_tests = attack.test_all()
                
        f = open('best_control_'+str(n_steps)+'.txt','w+')
        for key in best_control.keys():
            f.write(best_control[key]+'\n')
        f.write(str(best_loss))
        f.close()
        for key in best_control.keys():
            f = open('best_control_'+str(n_steps)+'_'+key+'.txt','w+')
            f.write(best_control[key])
            f.close()
        return self.control, step