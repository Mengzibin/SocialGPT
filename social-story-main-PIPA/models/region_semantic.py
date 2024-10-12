from models.segment_models.semgent_anything_model import SegmentAnything
from models.segment_models.semantic_segment_anything_model import SemanticSegment
from models.segment_models.edit_anything_model import EditAnything
from utils.util import str_n_list

def bbox_xywh_to_xyxy(bbox):
    new_bbox = []
    for i in range(len(bbox)):
        new_bbox.append([bbox[i][0],bbox[i][1],bbox[i][0]+bbox[i][2],bbox[i][1]+bbox[i][3]])
    return new_bbox

class RegionSemantic():
    def __init__(self, device, image_caption_model, region_classify_model='edit_anything', sam_arch='vit_b'):
        self.device = device
        self.sam_arch = sam_arch
        self.image_caption_model = image_caption_model
        self.region_classify_model = region_classify_model
        self.init_models()

    def init_models(self):
        self.segment_model = SegmentAnything(self.device, arch=self.sam_arch)
        if self.region_classify_model == 'ssa':
            self.semantic_segment_model = SemanticSegment(self.device)
        elif self.region_classify_model == 'edit_anything':
            self.edit_anything_model = EditAnything(self.image_caption_model)
            print('initalize edit anything model')
        else:
            raise ValueError("semantic_class_model must be 'ssa' or 'edit_anything'")

    def semantic_prompt_objects(self, anns):
        """
        anns: [{'class_name': 'person', 'bbox': [0.0, 0.0, 0.0, 0.0], 'size': [0, 0], 'stability_score': 0.0}, ...]
        objects: [{'name': 'O', 'bbox': [0.0, 0.0, 0.0, 0.0], 'class_name': "a man .."}]
        """
        sorted_annotations = anns
        objects = []
        for i in range(len(sorted_annotations)):
            objects.append({'name':'O'+str(i+1),'bbox':sorted_annotations[i]['bbox'],'class_name':sorted_annotations[i]['class_name']})
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return objects

    def semantic_prompt_people(self, anns):
        """
        anns: [{'class_name': 'person', 'bbox': [0.0, 0.0, 0.0, 0.0], 'size': [0, 0], 'stability_score': 0.0}, ...]
        objects: [{'name': 'O', 'bbox': [0.0, 0.0, 0.0, 0.0], 'class_name': "a man .."}]
        """
        sorted_annotations = sorted(anns, key=lambda x: x['area'], reverse=True)
        objects = []
        for i in range(len(sorted_annotations)):
            objects.append({'name':'O'+str(i+1),'bbox':sorted_annotations[i]['bbox'],'class_name':sorted_annotations[i]['class_name']})
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return objects

    def region_semantic(self, img_src, number, dir_result, region_classify_model='edit_anything'):
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("\nStep2, Semantic Prompt:")
        print('extract region segmentation with SAM model....\n')
        anns = self.segment_model.generate_mask(img_src,number,dir_result)
        print('finished...\n')
        if region_classify_model == 'ssa':
            print('generate region supervision with blip2 model....\n')
            anns_w_class = self.semantic_segment_model.semantic_class_w_mask(img_src, anns)
            print('finished...\n')
        elif region_classify_model == 'edit_anything':
            print('generate information (age and gender) caption of people in image...\n')
            anns_w_class = self.edit_anything_model.semantic_class_w_mask(img_src, anns)
            print('finished...\n')
        else:
            raise ValueError("semantic_class_model must be 'ssa' or 'edit_anything'")
        return self.semantic_prompt_objects(anns_w_class),anns_w_class
    
    def people_prompt(self, img_src, anns, region_classify_model='edit_anything', prompt="Question: How old is this person in the picture? Answer:"):
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print(prompt)
        if region_classify_model == 'ssa':
            print('generate region supervision with blip2 model....\n')
            anns_w_class = self.semantic_segment_model.semantic_class_w_mask(img_src, anns)
            print('finished...\n')
        elif region_classify_model == 'edit_anything':
            print('generate information (age and gender) caption of people in image...\n')
            anns_w_class = self.edit_anything_model.semantic_class_w_mask_prompt(img_src, anns, str_n_list(prompt,len(anns)))
            print('finished...\n')
        else:
            raise ValueError("semantic_class_model must be 'ssa' or 'edit_anything'")
        return self.semantic_prompt_people(anns_w_class)
    
    def region_semantic_debug(self, img_src):
        return "region_semantic_debug"

    def region_generate_people(self, img_src, people):
        bbox = []
        for i in range(len(people)):
            bbox.append(people[i]['bbox'])
        bbox = bbox_xywh_to_xyxy(bbox)
        anns = self.segment_model.generate_mask_bbox_high_level(img_src,bbox)
        return self.edit_anything_model.semantic_class_w_mask(img_src, anns)

