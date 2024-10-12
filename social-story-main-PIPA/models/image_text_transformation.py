from models.blip2_model import ImageCaptioning
from models.gpt_model import ImageToText
from models.controlnet_model import TextToImage
from models.region_semantic import RegionSemantic
from models.segment_models.semgent_anything_model import image_people_bbox,image_people_masks
from utils.util import read_image_width_height, display_images_and_text, resize_long_edge, scale_coordinate_newbbox,get_people_iou_region_semantic,get_semantic_region, check_tedious_objects,str_n_list
import argparse
from PIL import Image
import re
import base64
from io import BytesIO
import os

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

class ImageTextTransformation:
    def __init__(self, args):
        # Load your big model here
        self.args = args
        self.init_models()
        self.ref_image = None
    
    def init_models(self):
        openai_key = self.args.api_key
        print(self.args)
        print('\033[1;34m' + "Welcome to the social story toolbox...".center(50, '-') + '\033[0m')
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        print('\033[1;31m' + "This is time-consuming, please wait...".center(50, '-') + '\033[0m')
        self.image_caption_model = ImageCaptioning(device=self.args.image_caption_device, captioner_base_model=self.args.captioner_base_model)
        self.gpt_model = ImageToText(openai_key,self.args.gpt_version)
        self.controlnet_model = TextToImage(device=self.args.contolnet_device)
        self.region_semantic_model = RegionSemantic(device=self.args.semantic_segment_device, image_caption_model=self.image_caption_model, region_classify_model=self.args.region_classify_model, sam_arch=self.args.sam_arch)
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')

    def image_to_text(self, img_src, people):
        # the information to generate story based on the context
        self.ref_image = Image.open(img_src)
        # resize image to long edge 384
        self.ref_image = resize_long_edge(self.ref_image, 384)
        width, height = read_image_width_height(img_src)
        people = scale_coordinate_newbbox(width,height,people,self.ref_image.size[0],self.ref_image.size[1])
        if self.args.image_caption:
            image_caption = self.image_caption_model.image_caption(img_src)
            image_caption_scene = self.image_caption_model.image_caption_scene(img_src)
        else:
            image_caption = " "
        if self.args.semantic_segment:
            objects,anns_object = self.region_semantic_model.region_semantic(img_src,self.args.number,self.args.dir_result)
        else:
            objects = []
        real_people, final_objects, undetermined_people, anns_people = get_people_iou_region_semantic(people,objects,self.args.thresh_people, anns_object)
        if len(undetermined_people) > 0:
            anns_people_new = self.region_semantic_model.region_generate_people(img_src,undetermined_people)
            new_people = []
            for i in range(len(anns_people_new)):
                new_people.append({'name':undetermined_people[i]['name'],'bbox':anns_people_new[i]['bbox'],'class_name':anns_people_new[i]['class_name'],'segmentation':anns_people_new[i]['segmentation']})
            final_objects = check_tedious_objects(final_objects, new_people, self.args.thresh_people)
            for i in range(len(new_people)):
                real_people.append(new_people[i])
                anns_people.append(anns_people_new[i])
        
        image_people_masks(real_people,img_src,self.args.number,self.args.dir_result)
        people_age = self.region_semantic_model.people_prompt(img_src,anns_people)
        image_people_bbox(real_people,img_src,self.args.number,self.args.dir_result)
        people_gender = self.region_semantic_model.people_prompt(img_src,anns_people,prompt="Question: What gender is this person in the picture? Answer:")
        for i in range(len(real_people)):
            real_people[i]['age'] = people_age[i]['class_name']
            real_people[i]['gender'] = people_gender[i]['class_name']
        real_people = sorted(real_people,key = lambda i:int(re.findall(r'\d+',i['name'])[0]))
        region_semantic = get_semantic_region(final_objects, real_people)
        generated_text,question = self.gpt_model.paragraph_summary_with_gpt(image_caption, image_caption_scene, region_semantic, width, height)
        return generated_text, region_semantic, question

