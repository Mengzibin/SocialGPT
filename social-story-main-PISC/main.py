import argparse
import os
import cv2
from copy import deepcopy
from models.image_text_transformation import ImageTextTransformation
from models.text_relation_transformation import TextRelationTransformationPIPA,label_to_str
from utils.util import PeopleAndSample,get_label_from_answer,get_label_answer_from_last_sentence,get_label_statistic,print_label_category_statistic,create_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='PISC/image')
    parser.add_argument('--image_txt',default='relation_trainidx.txt')
    parser.add_argument('--dir_result',default='result',type=str)
    parser.add_argument('--api_key',type=str,default = 'openai-key',help='Your own api key of openai')
    parser.add_argument('--category_number',default=6,type=int,help='The number of category of social relationship in specific dataset')

    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
    parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, <6G GPU is not recommended>')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4' , 'gpt-3.5-turbo-0301' , 'gpt-3.5-turbo-0613'], default='gpt-4')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
    parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=True, help='Set this flag to True if you want to use semantic segmentation')
    parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_h', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
    parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip2', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
    parser.add_argument('--gpt_version_relation', choices=['gpt-3.5-turbo', 'gpt4' , 'gpt-3.5-turbo-0301' , 'gpt-3.5-turbo-0613'], default='gpt-4')

    parser.add_argument('--thresh_people',default=0.42)
    parser.add_argument('--index', default=0, type=int)

    args = parser.parse_args()

    People, Sample = PeopleAndSample(args.image_txt)
    
    processor = ImageTextTransformation(args)
    postcessor = TextRelationTransformationPIPA(args.api_key,gpt_version=args.gpt_version_relation)

    category = get_label_statistic(args.category_number)
    main_dir,sam_dir,relation_dir,answer_dir,result_category_dir,origin_example_dir = create_dir(args.dir_result)

    idx = 0
    sample = 0
    for image in People.keys():
        image_src = os.path.join(args.image_folder, image)
        prefix = image.split('.')[0]
        right = 0
        wrong = 0
        noanswer = 0
        if os.path.exists(image_src):
            args.number = idx + 1
            if idx >= args.index:
                generated_text,region_semantic,question = processor.image_to_text(image_src,People[image])
                f = open(os.path.join(main_dir,prefix+'_'+str(idx+1)+'.txt'),'w+')
                F = open(os.path.join(sam_dir,prefix+'_'+str(idx+1)+'.txt'),'w+')
                f.write(str(People[image])+'\n\n'+str(Sample[image])+'\n\n')
                f.write(region_semantic+'\n\n'+question+'\n\n'+generated_text+'\n\n')
                F.write(generated_text)

                R = open(os.path.join(relation_dir,prefix+'_'+str(idx+1)+'.txt'),'w+')
                H = open(os.path.join(answer_dir,prefix+'_'+str(idx+1)+'.txt'),'w+')
                C = open(os.path.join(result_category_dir,prefix+'_'+str(idx+1)+'.txt'),'w+')
                category_result = deepcopy(category)
                for i in range(len(Sample[image])):
                    sample = sample + 1
                    name1 = Sample[image][i]['R1']['name']
                    name2 = Sample[image][i]['R2']['name']
                    relation_str = label_to_str[Sample[image][i]['social_relationship']]
                    relation = Sample[image][i]['social_relationship']
                    answer,question = postcessor.text_to_relation_gpt_model(generated_text,name1,name2)
                    f.write(question+'\n'+answer+'\n\n')
                    R.write(answer+'\n\n'+relation_str+'\n\n')
                    f.write(relation_str+'\n\n')
                    label = get_label_answer_from_last_sentence(answer)
                    if relation in label:
                        right = right + 1
                        category_result[relation]['right'] += 1
                    elif len(label) > 0:
                        wrong = wrong + 1
                        category_result[relation]['wrong'] += 1
                    else:
                        noanswer = noanswer + 1
                        category_result[relation]['no'] += 1
                print_label_category_statistic(category_result,C)
                H.write(str(right)+' right answer\n'+str(wrong)+' wrong answer\n'+str(noanswer)+' no answer')
                H.close()
                R.close()
                C.close()

                img = cv2.imread(image_src)
                cv2.imwrite(os.path.join(origin_example_dir,prefix+'_'+str(idx+1)+'.jpg'),img)
                F.close()
                f.close()
            idx = idx + 1