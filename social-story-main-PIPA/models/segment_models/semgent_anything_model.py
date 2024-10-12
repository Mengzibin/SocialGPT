import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
from torchvision.ops.boxes import batched_nms
import imageio
from copy import deepcopy
import matplotlib.pyplot as pyplot
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from utils.util import resize_long_edge_cv2
from .amg import (MaskData,calculate_stability_score,batched_mask_to_box,is_box_near_crop_edge,uncrop_masks,uncrop_points,uncrop_boxes_xyxy,mask_to_rle_pytorch,coco_encode_rle,rle_to_mask,area_from_rle,box_xyxy_to_xywh,remove_small_regions)

color = [(0,0,255),(0,255,0),(255,0,0),(0,125,125),(125,125,0),
         (125,0,125),(60,60,60),(80,80,80),(255,255,0),(0,255,255),
         (255,0,255),(100,100,100),(120,120,120),(178,178,0),(200,200,0),
         (225,225,0),(0,178,178),(0,200,200),(178,0,178),(200,0,200),
         (50,50,0),(50,0,50),(0,50,50),(70,70,0),(0,70,70),(70,0,70),
         (90,90,0),(0,90,90),(90,0,90),(110,0,110),(0,110,110),(110,110,0),
         (140,140,0),(0,140,140),(140,0,140),(150,150,0),(0,150,150),(150,0,150),
         (160,0,160),(0,160,160),(160,160,0),(170,0,170),(170,170,0),(0,170,170),
         (190,190,0),(0,190,190),(190,0,190),(200,0,200),(0,200,200),(200,200,0),
         (0,0,255),(0,255,0),(255,0,0),(0,125,125),(125,125,0),
         (125,0,125),(60,60,60),(80,80,80),(255,255,0),(0,255,255),
         (255,0,255),(100,100,100),(120,120,120),(178,178,0),(200,200,0),
         (225,225,0),(0,178,178),(0,200,200),(178,0,178),(200,0,200),
         (50,50,0),(50,0,50),(0,50,50),(70,70,0),(0,70,70),(70,0,70),
         (90,90,0),(0,90,90),(90,0,90),(110,0,110),(0,110,110),(110,110,0),
         (140,140,0),(0,140,140),(140,0,140),(150,150,0),(0,150,150),(150,0,150),
         (160,0,160),(0,160,160),(160,160,0),(170,0,170),(170,170,0),(0,170,170),
         (190,190,0),(0,190,190),(190,0,190),(200,0,200),(0,200,200),(200,200,0),
         (0,0,255),(0,255,0),(255,0,0),(0,125,125),(125,125,0),
         (125,0,125),(60,60,60),(80,80,80),(255,255,0),(0,255,255),
         (255,0,255),(100,100,100),(120,120,120),(178,178,0),(200,200,0),
         (225,225,0),(0,178,178),(0,200,200),(178,0,178),(200,0,200),
         (50,50,0),(50,0,50),(0,50,50),(70,70,0),(0,70,70),(70,0,70),
         (90,90,0),(0,90,90),(90,0,90),(110,0,110),(0,110,110),(110,110,0),
         (140,140,0),(0,140,140),(140,0,140),(150,150,0),(0,150,150),(150,0,150),
         (160,0,160),(0,160,160),(160,160,0),(170,0,170),(170,170,0),(0,170,170),
         (190,190,0),(0,190,190),(190,0,190),(200,0,200),(0,200,200),(200,200,0)]

class SegmentPoint:
    def __init__(self, device, arch="vit_b"):
        self.device = device
        if arch=='vit_b':
            pretrained_weights="../pretrained_models/sam_vit_b_01ec64.pth"
        elif arch=='vit_l':
            pretrained_weights="../pretrained_models/sam_vit_l_0e2f7b.pth"
        elif arch=='vit_h':
            pretrained_weights="../pretrained_models/sam_vit_h_0e2f7b.pth"
        else:
            raise ValueError(f"arch {arch} not supported")
        self.model = self.initialize_model(arch, pretrained_weights)
    
    def initialize_model(self, arch, pretrained_weights):
        sam = sam_model_registry[arch](checkpoint=pretrained_weights)
        sam.to(device=self.device)
        mask_generator = SamPredictor(sam)
        return mask_generator

    def generate_mask(self, img_src, anns, number = 0):
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_long_edge_cv2(image, 384)
        anns = self.model.generate(image)
        return anns

class SegmentAnything:
    def __init__(self, device, arch="vit_b", 
                 pred_iou_thresh = 0.88,
                 points_per_batch: int = 64,
                 stability_score_thresh: float = 0.95,
                 stability_score_offset: float = 1.0,
                 box_nms_thresh: float = 0.7,
                 crop_n_layers: int = 0,
                 crop_nms_thresh: float = 0.7,
                 crop_overlap_ratio: float = 512 / 1500,
                 crop_n_points_downscale_factor: int = 1,
                 min_mask_region_area: int = 0,
                 output_mode: str = "binary_mask"):
        self.device = device
        if arch=='vit_b':
            pretrained_weights="../pretrained_models/sam_vit_b_01ec64.pth"
        elif arch=='vit_l':
            pretrained_weights="../pretrained_models/sam_vit_l_0e2f7b.pth"
        elif arch=='vit_h':
            pretrained_weights="../pretrained_models/sam_vit_h_4b8939.pth"
        else:
            raise ValueError(f"arch {arch} not supported")
        self.model = self.initialize_model(arch, pretrained_weights)
        sam_point_bbox = sam_model_registry['vit_h'](checkpoint="../pretrained_models/sam_vit_h_4b8939.pth")
        sam_point_bbox.to(device=self.device)
        self.mask_model = SamPredictor(sam_point_bbox)

        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
    
    def initialize_model(self, arch, pretrained_weights):
        sam = sam_model_registry[arch](checkpoint=pretrained_weights)
        sam.to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator

    def generate_mask(self, img_src, number = 0, dir_result = 'result'):
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_long_edge_cv2(image, 384)
        imageori = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        anns_edit_anything = self.model.generate(image)
        anns_high_level = self.generate_mask_high_level(anns_edit_anything,image)
        image_original_mask(anns_high_level,imageori,number,dir_result)
        image_original(anns_high_level, imageori, number, dir_result)
        return anns_high_level
    
    def generate_mask_bbox_high_level(self,img_src,bbox):
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_long_edge_cv2(image, 384)
        orig_h,orig_w = image.shape[:2]
        self.mask_model.set_image(image)
        transformed_boxes = self.mask_model.transform.apply_boxes_torch(torch.tensor(bbox,device=self.mask_model.device), image.shape[:2])
        masks, iou_preds, _ = self.mask_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
            return_logits=True
        )
        masks = masks > 0.0
        crop_box = [0,0,image.shape[1],image.shape[0]]
        anns = self.get_anns_bbox_without_points(iou_preds,masks,crop_box,orig_w,orig_h)
        return anns

    def generate_mask_high_level(self,anns,image):
        points = []
        labels = []
        for i in range(len(anns)):
            x = int((anns[i]['bbox'][0]*2+anns[i]['bbox'][2])/2)
            y = int((anns[i]['bbox'][1]*2+anns[i]['bbox'][3])/2)
            points.append([x,y])
            labels.append(1)
        points,labels = torch.tensor(points),torch.tensor(labels)
        points_,labels = points.reshape(points.shape[0],1,points.shape[1]),labels.reshape(labels.shape[0],1)
        self.mask_model.set_image(image)
        transformed_coords = self.mask_model.transform.apply_coords_torch(torch.tensor(points_,device=self.mask_model.device), image.shape[:2])
        crop_box = [0,0,image.shape[1],image.shape[0]]
        orig_h,orig_w = image.shape[:2]
        masks, iou_preds, _ = self.mask_model.predict_torch(
            point_coords=transformed_coords,
            point_labels=labels,
            multimask_output=True,
            return_logits=True
        )
        data = MaskData(
            masks=masks[:,2,:,:],
            iou_preds=iou_preds[:,2],
            points=torch.as_tensor(np.array(points)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh - 0.03
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], 0.0, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh - 0.03
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > 0.0
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])

        self.mask_model.reset_image()

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"])),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        area = int(orig_h * orig_w * 0.015)
        keep_mask = ((data['boxes'][:,2] - data['boxes'][:,0]) * (data['boxes'][:,3] - data['boxes'][:,1])) > area
        data.filter(keep_mask)

        data.to_numpy()
        mask_data = data
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)
        return curr_anns
    
    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def get_anns_bbox_without_points(self, iou_preds, masks, crop_box, orig_w, orig_h):
        data = MaskData(
            masks=masks[:,2,:,:],
            iou_preds=iou_preds[:,2],
        )
        del masks

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > 0.0
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])

        self.mask_model.reset_image()

        # keep_by_nms = batched_nms(
        #     data["boxes"].float(),
        #     data["iou_preds"],
        #     torch.zeros(len(data["boxes"])),  # categories
        #     iou_threshold=self.box_nms_thresh,
        # )
        # data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        data.to_numpy()
        mask_data = data
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)
        return curr_anns

def image_fun(anns,image_list,number):
    for i in range(len(anns)): 
        for j in range(len(anns[i])):
            out_image(anns[i][j]['segmentation'],j,anns[i][j]['point_coords'][0],anns[i][j]['bbox'], number,i)
    for i in range(len(anns)):
        anns = sorted(anns[i], key=(lambda x: x['area']), reverse=True)
        for k in range(0,min(10,len(anns[i]))):
            cv2.rectangle(image_list[i],(int(anns[k]['bbox'][0]),int(anns[k]['bbox'][1])),(int(anns[k]['bbox'][0])+int(anns[k]['bbox'][2]),int(anns[k]['bbox'][1])+int(anns[k]['bbox'][3])),color[k],2)
        cv2.imwrite(str(number)+'_'+str(i)+'.jpg',image_list[i])

def out_image(array,idx,point,bbox,number,number2):
    bbox = list(np.array(bbox).astype(int))
    length = array.shape[0]
    width = array.shape[1]
    matrix = np.zeros((length,width),dtype=np.int_)
    for i in range(0,length):
        for j in range(0,width):
            if array[i][j] == False:
                matrix[i][j] = 0
            elif array[i][j] == True:
                matrix[i][j] = 1
    matrix[int(point[1])][int(point[0])] = 1
    matrix[int(point[1])][int(point[0])+1] = 0
    matrix[int(point[1])+1][int(point[0])] = 0
    matrix[int(point[1])-1][int(point[0])] = 0
    matrix[int(point[1])][int(point[0])-1] = 0
    matrix[int(point[1])+1][int(point[0])+1] = 0
    matrix[int(point[1])-1][int(point[0])+1] = 0
    matrix[int(point[1])+1][int(point[0])-1] = 0
    matrix[int(point[1])-1][int(point[0])-1] = 0
    matrix[int(point[1])+2][int(point[0])+2] = 1
    matrix[int(point[1])+2][int(point[0])-2] = 1
    matrix[int(point[1])-2][int(point[0])+2] = 1
    matrix[int(point[1])-2][int(point[0])-2] = 1
    for i in range(0,bbox[3]):
        matrix[bbox[1]+i][bbox[0]] = 1
        matrix[bbox[1]+i][bbox[0]+bbox[2]] = 1
    for i in range(0,bbox[2]):
        matrix[bbox[1]][bbox[0]+i] = 1
        matrix[bbox[1]+bbox[3]][bbox[0]+i] = 1
    out_img(matrix,idx,number,number2)

def out_img(data,idx,number,number2):
    data = (data * 255.0).astype('uint8')
    new_im = Image.fromarray(data)
    imageio.imsave('mask/'+str(number)+'_'+str(number2)+'_'+str(idx)+'.jpg', new_im)

def image_original(anns,imageori,number,result_dir):
    # for i in range(len(anns)): 
    #     out_image_original(anns[i]['segmentation'],i,anns[i]['point_coords'][0],anns[i]['bbox'], number)
    for k in range(len(anns)):
        cv2.rectangle(imageori,(int(anns[k]['bbox'][0]),int(anns[k]['bbox'][1])),(int(anns[k]['bbox'][0])+int(anns[k]['bbox'][2]),int(anns[k]['bbox'][1])+int(anns[k]['bbox'][3])),color[k],2)
    story_example_dir = os.path.join(result_dir,'story_example')
    if not os.path.exists(story_example_dir):
        os.makedirs(story_example_dir)
    cv2.imwrite(os.path.join(story_example_dir,str(number)+'.jpg'),imageori)

def image_people_bbox(people,img_src,number,dir_result):
    image = cv2.imread(img_src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_long_edge_cv2(image, 384)
    imageori = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # for i in range(len(anns)): 
    #     out_image_original(anns[i]['segmentation'],i,anns[i]['point_coords'][0],anns[i]['bbox'], number)
    for k in range(len(people)):
        cv2.rectangle(imageori,(int(people[k]['bbox'][0]),int(people[k]['bbox'][1])),(int(people[k]['bbox'][0])+int(people[k]['bbox'][2]),int(people[k]['bbox'][1])+int(people[k]['bbox'][3])),color[k],2)
    story_example_people_dir = os.path.join(dir_result,'story_example_people')
    if not os.path.exists(story_example_people_dir):
        os.makedirs(story_example_people_dir)
    cv2.imwrite(os.path.join(story_example_people_dir,str(number)+'.jpg'),imageori)

def out_image_original(array,idx,point,bbox,number):
    bbox = list(np.array(bbox).astype(int))
    length = array.shape[0]
    width = array.shape[1]
    matrix = np.zeros((length,width),dtype=np.int_)
    for i in range(0,length):
        for j in range(0,width):
            if array[i][j] == False:
                matrix[i][j] = 0
            elif array[i][j] == True:
                matrix[i][j] = 1
    matrix[int(point[1])][int(point[0])] = 1
    matrix[int(point[1])][int(point[0])+1] = 0
    matrix[int(point[1])+1][int(point[0])] = 0
    matrix[int(point[1])-1][int(point[0])] = 0
    matrix[int(point[1])][int(point[0])-1] = 0
    matrix[int(point[1])+1][int(point[0])+1] = 0
    matrix[int(point[1])-1][int(point[0])+1] = 0
    matrix[int(point[1])+1][int(point[0])-1] = 0
    matrix[int(point[1])-1][int(point[0])-1] = 0
    matrix[int(point[1])+2][int(point[0])+2] = 1
    matrix[int(point[1])+2][int(point[0])-2] = 1
    matrix[int(point[1])-2][int(point[0])+2] = 1
    matrix[int(point[1])-2][int(point[0])-2] = 1
    for i in range(0,bbox[3]):
        matrix[bbox[1]+i][bbox[0]] = 1
        matrix[bbox[1]+i][bbox[0]+bbox[2]] = 1
    for i in range(0,bbox[2]):
        matrix[bbox[1]][bbox[0]+i] = 1
        matrix[bbox[1]+bbox[3]][bbox[0]+i] = 1
    out_img_original(matrix,idx,number)

def out_img_original(data,idx,number):
    data = (data * 255.0).astype('uint8')
    new_im = Image.fromarray(data)
    imageio.imsave('mask/'+str(number)+'_'+str(idx)+'.jpg', new_im)

def show_anns(anns):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.55]])
        img[m] = color_mask
    ax.imshow(img)

def image_people_masks(people,img_src,number,dir_result):
    image = cv2.imread(img_src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_long_edge_cv2(image, 384)
    imageori = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.imshow(image)
    show_anns(people)
    plt.axis('off')
    story_example_people_dir = os.path.join(dir_result,'story_example_mask_people')
    if not os.path.exists(story_example_people_dir):
        os.makedirs(story_example_people_dir)
    plt.savefig(os.path.join(story_example_people_dir,str(number)+'.jpg'),bbox_inches='tight',pad_inches=0)

def image_original_mask(anns,imageori,number,result_dir):
    # for i in range(len(anns)): 
    #     out_image_original(anns[i]['segmentation'],i,anns[i]['point_coords'][0],anns[i]['bbox'], number)
    plt.figure()
    plt.imshow(imageori)
    show_anns(anns)
    plt.axis('off')
    story_example_dir = os.path.join(result_dir,'story_example_mask')
    if not os.path.exists(story_example_dir):
        os.makedirs(story_example_dir)
    plt.savefig(os.path.join(story_example_dir,str(number)+'.jpg'),bbox_inches='tight',pad_inches=0)