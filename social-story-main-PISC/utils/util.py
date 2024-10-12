from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import textwrap
from copy import deepcopy
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from models.text_relation_transformation import str_to_label

def read_image_width_height(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height

def resize_long_edge(image, target_size=384):
    # Calculate the aspect ratio
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    # Determine the new dimensions
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def resize_long_edge_cv2(image, target_size=384):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    else:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def display_images_and_text(source_image_path, generated_image, generated_paragraph, outfile_name):
    source_image = Image.open(source_image_path)
    # Create a new image that can fit the images and the text
    width = source_image.width + generated_image.width
    height = max(source_image.height, generated_image.height)
    new_image = Image.new("RGB", (width, height + 150), "white")

    # Paste the source image and the generated image onto the new image
    new_image.paste(source_image, (0, 0))
    new_image.paste(generated_image, (source_image.width, 0))

    # Write the generated paragraph onto the new image
    draw = ImageDraw.Draw(new_image)
    # font_size = 12
    # font = ImageFont.load_default().font_variant(size=font_size)
    font_path = os.path.join('C:\Windows\Fonts\Times New Roman.ttf')

    # Wrap the text for better display
    wrapped_text = textwrap.wrap(generated_paragraph, width=170)
    # Draw each line of wrapped text
    line_spacing = 18
    y_offset = 0
    for line in wrapped_text:
        draw.text((0, height + y_offset), line,  fill="black")
        y_offset += line_spacing

    # Show the final image
    # new_image.show()
    new_image.save(outfile_name)
    return 1


def extract_nouns_nltk(paragraph):
    words = word_tokenize(paragraph)
    pos_tags = pos_tag(words)
    nouns = [word for word, tag in pos_tags if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
    return nouns

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

def scale_coordinate_newbbox(width,height,people,new_width,new_height):
    for i in range(len(people)):
        coor = people[i]['bbox']
        scale_x = new_width / width
        scale_y = new_height / height
        x1,x2 = int(scale_x*coor[0]),int(scale_x*coor[2])
        y1,y2 = int(scale_y*coor[1]),int(scale_y*coor[3])
        people[i]['bbox'] = [x1,y1,x2-x1,y2-y1]
    return people

def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    get iou confidence of different bboxes
    '''
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)

def get_people_iou_region_semantic(people,objects,thresh,anns):
    new_people = []
    undetermined_people = []
    name_list = []
    new_objects = []
    new_anns_people = []
    objects_ori = deepcopy(objects)
    anns_pop = deepcopy(anns)
    for i in range(len(people)):
        if len(objects) > 0:
            iou = []
            v = people[i]['bbox']
            for j in range(len(objects)):
                iou.append(bb_overlab(objects[j]['bbox'][0],objects[j]['bbox'][1],
                                                            objects[j]['bbox'][2],objects[j]['bbox'][3],
                                                            v[0],v[1],v[2],v[3]))
            if max(iou) > thresh:
                name_list.append(objects[iou.index(max(iou))]['name'])
                new_people.append({'name':people[i]['name'],'bbox':objects[iou.index(max(iou))]['bbox'],'class_name':objects[iou.index(max(iou))]['class_name'],'segmentation':anns_pop[iou.index(max(iou))]['segmentation']})
                assert anns_pop[iou.index(max(iou))]['bbox'] == objects[iou.index(max(iou))]['bbox']
                objects.pop(iou.index(max(iou)))
                anns_pop.pop(iou.index(max(iou)))
            else:
                undetermined_people.append(people[i])
        else:
            undetermined_people.append(people[i])
    new_anns_people = [anns[i] for i in range(len(objects_ori)) if objects_ori[i]['name'] in name_list]
    new_objects = [k for k in objects if k['name'] not in name_list]
    for i in range(len(new_objects)):
        new_objects[i]['name'] = 'O'+str(i+1)
    return new_people,new_objects,undetermined_people,new_anns_people

def get_semantic_region(objects,people):
    object_semantic = ""
    people_semantic = ""
    for i in range(len(objects)):
        object_semantic += '{<symbol>:[' + objects[i]['name'] + '] , <coordinate>:' + str(objects[i]['bbox']) + ' , <caption>:[' + objects[i]['class_name'] + ']}; '
    for i in range(len(people)):
        people_semantic += '{<symbol>:[' + people[i]['name'] + '] , <coordinate>:' + str(people[i]['bbox']) + ' , <caption>:[' + people[i]['class_name'] + '] , <age>:[' + people[i]['age'] + '] , <gender>:[' + people[i]['gender'] + ']}; '
    region_semantic = people_semantic + object_semantic
    return region_semantic

def check_tedious_objects(objects,people,thresh):
    name_list = []
    for i in range(len(people)):
        if len(objects) > 0:
            iou = []
            v = people[i]['bbox']
            for j in range(len(objects)):
                iou.append(bb_overlab(objects[j]['bbox'][0],objects[j]['bbox'][1],
                                                            objects[j]['bbox'][2],objects[j]['bbox'][3],
                                                            v[0],v[1],v[2],v[3]))
            if max(iou) > thresh:
                name_list.append(objects[iou.index(max(iou))]['name'])
    new_objects = [k for k in objects if k['name'] not in name_list]
    for i in range(len(new_objects)):
        new_objects[i]['name'] = 'O'+str(i+1)
    return new_objects

def get_label_from_answer(answer):
    label = []
    result = answer[answer.rfind('Result')+len('Result'):]
    for key in str_to_label.keys():
        if key in result:
            label.append(str_to_label[key])
    return label

def get_label_answer_from_last_sentence(answer):
    sentence = answer.split('.')[0]
    label = []
    for key in str_to_label.keys():
        if key in sentence:
            label.append(str_to_label[key])
    return label

def get_label_statistic(number):
    dic = {}
    for i in range(number):
        dic[i] = {}
        dic[i]['right'] = 0
        dic[i]['wrong'] = 0
        dic[i]['no'] = 0
    return dic

def print_label_category_statistic(category_result,F):
    for key in category_result.keys():
        F.write(str(key)+' '+str(category_result[key]['right'])+' '+str(category_result[key]['wrong'])+' '+str(category_result[key]['no'])+'\n')

def str_n_list(str,n):
    lis = []
    for i in range(n):
        lis.append(str)
    return lis

def create_dir(dir_result):
    main_dir = os.path.join(dir_result,'caption/caption_main')
    sam_dir = os.path.join(dir_result,'caption/caption_sam')
    relation_dir = os.path.join(dir_result,'caption/caption_relation')
    answer_dir = os.path.join(dir_result,'caption/caption_answer')
    result_category_dir = os.path.join(dir_result,'caption/caption_result_category')
    origin_example_dir = os.path.join(dir_result,'origin_example')

    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    if not os.path.exists(sam_dir):
        os.makedirs(sam_dir)
    if not os.path.exists(relation_dir):
        os.makedirs(relation_dir)
    if not os.path.exists(answer_dir):
        os.makedirs(answer_dir)
    if not os.path.exists(result_category_dir):
        os.makedirs(result_category_dir)
    if not os.path.exists(origin_example_dir):
        os.makedirs(origin_example_dir)
    
    return main_dir,sam_dir,relation_dir,answer_dir,result_category_dir,origin_example_dir