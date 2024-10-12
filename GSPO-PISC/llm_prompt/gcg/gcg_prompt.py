from base import StoryPrompt
from base import PromptManager
from base import MultiPromptStory
from base import get_embeddings
from base import get_embedding_matrix
import gc
import torch
from tqdm.auto import tqdm
import random
import numpy as np
import torch.nn as nn

def token_gradients(model, 
        controls, 
        tokenizer,
        input_ids, 
        system_prompt_slice, 
        prelude_slice, 
        relationship_slice, 
        example_slice, 
        target_slice, 
        loss_slice
    ):
    controls_toks = {}
    controls_toks['system_prompt'] = [torch.tensor(tokenizer(controls['system_prompt'][i]).input_ids) for i in range(len(controls['system_prompt']))]
    controls_toks['prelude'] = [torch.tensor(tokenizer(controls['prelude'][i]).input_ids) for i in range(len(controls['prelude']))]
    controls_toks['relationship'] = [torch.tensor(tokenizer(controls['relationship'][i]).input_ids) for i in range(len(controls['relationship']))]
    controls_toks['example'] = [torch.tensor(tokenizer(controls['example'][i]).input_ids) for i in range(len(controls['example']))]

    new_pad_tok = 0
    while new_pad_tok in input_ids or any([new_pad_tok in ids for key in controls_toks.keys() for ids in controls_toks[key]]):
        new_pad_tok += 1

    full_embeds_ids = {}
    attn_masks = {}
    for key in controls_toks.keys():
        new_embeds_ids = []
        start = int(eval(key+'_slice').start) if eval(key+'_slice').start != 1 else 0
        stop = int(eval(key+'_slice').stop)
        for i in range(len(controls_toks[key])):
            new_embeds_ids.append(
                torch.cat(
                    [
                        input_ids[:start],
                        controls_toks[key][i].to(model.device),
                        input_ids[stop:]
                    ],
                    dim=0
                )
            )
        maxlen = max([new_embeds_ids[i].shape[0] for i in range(len(new_embeds_ids))])
        final_embeds_ids = []
        for i in range(len(new_embeds_ids)):
            num_specific_elements = maxlen - new_embeds_ids[i].shape[0]
            specific_elements = torch.full((num_specific_elements,),new_pad_tok)
            expanded_vector = torch.cat([specific_elements.to(model.device),new_embeds_ids[i]])
            final_embeds_ids.append(list(expanded_vector))
        full_embeds_ids[key] = torch.tensor(final_embeds_ids).to(model.device)
        attn_masks[key] = (torch.ne(full_embeds_ids[key], new_pad_tok)).type(torch.int64)
    del new_embeds_ids,num_specific_elements
    
    one_hot_grad_clone = {}
    for key in full_embeds_ids.keys():
        one_hot = torch.zeros(
            1,
            full_embeds_ids[key].shape[0],
            device=model.device,
            dtype=torch.float16
        )
        one_hot.scatter_(
            1,
            get_token_single(full_embeds_ids[key].shape[0]).to(model.device),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=torch.float16)
        ).reshape(full_embeds_ids[key].shape[0])
        one_hot.requires_grad_()
        embeds_ids = get_embeddings(model, full_embeds_ids[key]).to(get_embedding_matrix(model).dtype)
        different_value = full_embeds_ids[key].shape[1] - input_ids.shape[0]
        input_embeds = torch.matmul(one_hot,embeds_ids.view(embeds_ids.shape[0],-1).detach()).view(1,embeds_ids.shape[1],embeds_ids.shape[2])
        
        print(f'input_embeds shape : {input_embeds.shape}')
        if input_embeds.shape[1] > 2320:
            one_hot_grad_clone[key] = torch.zeros_like(one_hot).to(model.device)
        else:
            attn_mask = one_hot @ attn_masks[key].to(torch.float16)
            logits = model(inputs_embeds=input_embeds, attention_mask=attn_mask).logits
            targets = input_ids[target_slice]
            loss = nn.CrossEntropyLoss()(logits[0,slice(loss_slice.start+different_value,loss_slice.stop+different_value),:],targets)
            loss.backward()
            one_hot_grad_clone[key] = one_hot.grad.clone()
            print(f'one_hot_grad_clone : {one_hot.grad.clone().device}')
    return one_hot_grad_clone

def get_token_single(num_prompt):
    lis = []
    lis.append(random.randint(0,num_prompt-1))
    return torch.tensor(lis).reshape(1,1)

class GCGStoryPrompt(StoryPrompt):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def grad(self,model):
        input_ids_without_pad_tok,system_prompt_slice,prelude_slice,relationship_slice,example_slice,target_slice,loss_slice = self._get_input_ids_without_pad_tok()
        return token_gradients(
            model,
            self.controls,
            self.tokenizer,
            input_ids_without_pad_tok.to(model.device),
            system_prompt_slice,
            prelude_slice,
            relationship_slice,
            example_slice,
            target_slice,
            loss_slice,
        )
    
class GCGPromptManager(PromptManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=3, temp=1, allow_non_ascii=True):
        control_str = self.control_str
        new_controls = {}
        for key in grad.keys():
            if not allow_non_ascii:
                grad[key][:,self._nonascii_toks.to(grad[key].device)] = np.infty
            top_indice = (-grad[key]).topk(topk,dim=1).indices.to(torch.int64)
            new_controls[key] = [self.controls[key][i] for i in top_indice[0]]
        final_controls = {}
        key_list = list(new_controls.keys())
        for key in new_controls.keys():
            final_controls[key] = []

        for key in new_controls.keys():
            for i in range(len(new_controls[key])):
                final_controls[key].append(new_controls[key][i])
            for key_ano in new_controls.keys():
                if key_ano != key:
                    final_controls[key_ano] += [control_str[key_ano]] * len(new_controls[key_ano])

        assert len(final_controls[key_list[0]]) == len(final_controls[key_list[1]]) == len(final_controls[key_list[2]]) == len(final_controls[key_list[3]])
        assert len(final_controls[key_list[0]]) == batch_size
        return final_controls
    
    def controls_toks_repeat(self,controls_toks,batch_size):
        new_controls_toks = {}
        for key in controls_toks:
            new_controls_toks[key] = controls_toks[key].repeat(batch_size,1)
        return new_controls_toks

class GCGMultiPromptStory(MultiPromptStory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, 
             batch_size=100, 
             topk=3, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             verbose=False, 
             filter_cand=True):
        opt_only = False
        main_device = self.models[0].device
        control_cands = []
        for j, worker in enumerate(self.workers):
            worker(self.prompts[j],"grad",worker.model)
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get()
            new_grad = self.new_grad_to_model(new_grad,main_device)
            new_grad = self.new_grad_norm(new_grad)
            if grad is None:
                grad = self.zeros_like_new_grad(new_grad)
            if self.judge_grad_new_grad_shape_not_equal(grad,new_grad):
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad,batch_size,topk,temp,allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1,control_cand,filter_cand=filter_cand,curr_control=self.control_str))
                grad = new_grad
            else:
                grad = self.add_new_grad_and_grad(new_grad,grad)
        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad,batch_size,topk,temp,allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad,control_cand; gc.collect()

        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model,cand, return_ids=True)
                    logits,ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit,id).mean(dim=-1).to(main_device)
                        for k,(logit,id) in enumerate(zip(logits,ids))
                    ])
                    del logits,ids;gc.collect()

                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control = {}
            for key in control_cands[model_idx].keys():
                next_control[key] = control_cands[model_idx][key][batch_idx]
            cand_loss = loss[min_idx]
        
        del control_cands,loss;gc.collect()
        torch.cuda.empty_cache()
        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
    

    def new_grad_to_model(self,new_grad,main_device):
        new_grads = {}
        for key in new_grad.keys():
            new_grads[key] = new_grad[key].to(main_device)
        return new_grads

    def new_grad_norm(self,new_grad):
        new_grads = {}
        for key in new_grad.keys():
            new_grads[key] = new_grad[key] / new_grad[key].norm(dim=-1, keepdim=True)
        return new_grads
    
    def zeros_like_new_grad(self,new_grad):
        zero_grad = {}
        for key in new_grad.keys():
            zero_grad[key] = torch.zeros_like(new_grad[key])
        return zero_grad

    def judge_grad_new_grad_shape_not_equal(self,grad,new_grad):
        for key in grad.keys():
            if new_grad[key].shape != grad[key].shape:
                return True
        return False

    def add_new_grad_and_grad(self,new_grad,grad):
        grad_final = {}
        for key in new_grad.keys():
            grad_final[key] = new_grad[key] + grad[key]
        return grad_final