import os
import re
import random
from fastchat.model import get_conversation_template

str_to_label = {'friends':0,'family-members':1,'couple':2,'professional':3,'commercial':4,'no-relationship':5}

label_to_str = {0:'friends',1:'family-members',2:'couple',3:'professional',4:'commercial',5:'no-relationship'}

def get_txt_file_from_dir(dir):
    for root,_,files in os.walk(dir):
        txt_list = []
        files = sorted(files,key = lambda i:int(re.findall(r'\d+',i)[0]))
        for file in files:
            path = os.path.join(root,file)
            F = open(path,'r')
            string = ''
            txt = F.readlines()
            for i in range(len(txt)):
                string += txt[i]
            txt_list.append(string)
            F.close()
    return txt_list

def get_story(dir):
    for root,_,files in os.walk(dir):
        txt_dict = {}
        files = sorted(files,key = lambda i:int(re.findall(r'\d+',i)[0]))
        for file in files:
            path = os.path.join(root,file)
            F = open(path,'r')
            string = ''
            txt = F.readlines()
            for i in range(len(txt)):
                string += txt[i]
            txt_dict[file.split('.')[0].split('_')[0]] = string
            F.close()
    return txt_dict

def get_data_to_prompt(params):
    example_dir = os.path.join(params.sam4story,params.example_dir)
    relationship_dir = os.path.join(params.sam4story,params.relationship_dir)
    prelude_dir = os.path.join(params.sam4story,params.prelude_dir)
    story_dir = os.path.join(params.sam4story,params.story_dir)
    system_prompt_dir = os.path.join(params.sam4story,params.system_prompt_dir)

    example = get_txt_file_from_dir(example_dir)
    relationship = get_txt_file_from_dir(relationship_dir)
    prelude = get_txt_file_from_dir(prelude_dir)
    story = get_story(story_dir)
    system_prompt = get_txt_file_from_dir(system_prompt_dir)

    return example,relationship,prelude,story,system_prompt

def Peoplein(list,coordinate):
    if len(list) == 0:
        return False, None
    else:
        for i in range(len(list)):
            for k,v in list[i].items():
                if coordinate == v:
                    return True, list[i]['name']
    return False, None

def PeopleAndSample(file = 'domain_train.txt'):
    Sample = {}
    People = {}
    with open(file, 'r') as fin:
        for line in fin:
            data = line.split()
            if data[0] not in Sample.keys():
                People[data[0]] = []
                Sample[data[0]] = []
            length = len(People[data[0]])
            V1 = [int(data[1]),int(data[2]),int(data[3]),int(data[4])]
            V2 = [int(data[5]),int(data[6]),int(data[7]),int(data[8])]
            relationship = int(data[9])
            J1,P1 = Peoplein(People[data[0]],V1)
            J2,P2 = Peoplein(People[data[0]],V2)
            if not J1:
                People[data[0]].append({'name':'P'+str(length+1),'bbox':V1})
                P1 = 'P'+str(length+1)
                length = length + 1
            if not J2:
                People[data[0]].append({'name':'P'+str(length+1),'bbox':V2})
                P2 = 'P'+str(length+1)
                length = length + 1
            Sample[data[0]].append({'R1':{'name':P1,'bbox':V1},'R2':{'name':P2,'bbox':V2},"social_relationship":relationship})
    return People, Sample

def get_targets_and_stories(params):
    example,relationship,prelude,story,system_prompt = get_data_to_prompt(params)
    People,Sample = PeopleAndSample(params.image_txt)
    prompt = {
        'example':example,
        'relationship':relationship,
        'prelude':prelude,
        'system_prompt':system_prompt
    }
    question_list = []
    answer_list = []
    relation_list = []
    for image in People.keys():
        if image.split('.')[0] not in story.keys():
            continue
        image_story = story[image.split('.')[0]]
        for i in range(len(Sample[image])):
            name1 = Sample[image][i]['R1']['name']
            name2 = Sample[image][i]['R2']['name']
            relation_str = label_to_str[Sample[image][i]['social_relationship']]
            answer = f'[The final answer is <{relation_str}>]'
            question = {
                'story':image_story,
                'R1':name1,
                'R2':name2
            }
            question_list.append(question)
            answer_list.append(answer)
            relation_list.append(relation_str)
    return prompt,question_list,answer_list,relation_list

def get_story_test(params):
    story_dir = os.path.join(params.sam4story,params.story_dir_test)
    story = get_story(story_dir)
    return story

def get_targets_and_questions_of_test(params):
    story = get_story_test(params)
    People,Sample = PeopleAndSample(params.image_txt_test)
    question_list = []
    answer_list = []
    relationship_list = []
    for image in People.keys():
        if image.split('.')[0] not in story.keys():
            continue
        image_story = story[image.split('.')[0]]
        for i in range(len(Sample[image])):
            name1 = Sample[image][i]['R1']['name']
            name2 = Sample[image][i]['R2']['name']
            relation_str = label_to_str[Sample[image][i]['social_relationship']]
            answer = f'[The final answer is <{relation_str}>]'
            question = {
                'story':image_story,
                'R1':name1,
                'R2':name2
            }
            question_list.append(question)
            answer_list.append(answer)
            relationship_list.append(relation_str)
    return question_list,answer_list,relationship_list

def prompt_relation(example,relationship,prelude,question,system_prompt):
    prompt =  prelude+'\n\n'+relationship+'\n\n'+'****************************************************************\nExample:\n'+example+'\n****************************************************************\n\n[1. image description]:\n'+question['story']+'\n\n[2. Question]: \nWhat are the most likely social relationships between '+question['R1']+' and '+question['R2']+'? Choose only one from {<friends>, <family-members>, <couple>, <professional>, <commercial>, <no-relationship>}.\n\n[3. Answer]:'
    return prompt

def pad_string_with_spaces(input_string, target_length):
    if len(input_string) >= target_length:
        return input_string
    else:
        spaces_to_add = target_length - len(input_string)
        padded_string = input_string + ' ' * spaces_to_add
        return padded_string

def get_random_idx(relationships_total):
    relationship_idx = {}
    idx = []
    for key in str_to_label.keys():
        relationship_idx[key] = []
    for i in range(len(relationships_total)):
        relationship_idx[relationships_total[i]].append(i)
    for key in relationship_idx.keys():
        idx.append(random.choice(relationship_idx[key]))
    return idx

def get_prompt_from_best(file):
    line_list = []
    with open(file, 'r') as f:
        idx = 0
        relationship = ""
        example = ""
        system_prompt = ""
        line_1 = 18
        line_2 = 25
        for line in f.readlines():
            if idx == 0:
                prelude = line.strip()
            if idx > 0 and idx <= line_1:
                if idx == line_1:
                    relationship += line.strip()
                else:
                    relationship += line
            if idx > line_1 and idx <= line_2:
                if idx == line_2:
                    example += line.strip()
                else:
                    example += line
            if idx > line_2 and idx <= 32:
                if idx == 32:
                    system_prompt += line.strip()
                else:
                    system_prompt += line
            idx = idx + 1
    print(f'{system_prompt}\n')
    print(f'{prelude}\n')
    print(f'{relationship}\n')
    print(f'{example}')
    return system_prompt,prelude,relationship,example