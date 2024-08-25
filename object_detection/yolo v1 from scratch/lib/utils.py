import torch
from torch import Tensor
import typing as tt
from enum import Enum

class BoxShape(Enum):
    CORNERS = "corners" # (x1,y1,x2,y2)
    MIDPOINT = "midpoint" # (x1,y1,w,h)

def get_coords(boxes:Tensor,box_shape:str)->tt.Tuple[Tensor,Tensor,Tensor,Tensor]:

    try:
        shape = BoxShape(box_shape)
    except ValueError:
        raise ValueError(f"Uknown shape type : {box_shape} ") 

    if shape == BoxShape.CORNERS:
        boxes_x1 =boxes[...,0:1] # keep the shape : (N ,1 )
        boxes_y1 =boxes[...,1:2]
        boxes_x2 =boxes[...,2:3]
        boxes_y2 =boxes[...,3:4]
    else:
        # shape is midpoint (x_c,y_c,w,h)
        # we do coord = center - (w or h)/2
        boxes_x1 =boxes[...,0:1]-boxes[...,2:3]/2 # keep the shape : (N ,1 )
        boxes_y1 =boxes[...,1:2]-boxes[...,3:4]/2
        boxes_x2 =boxes[...,0:1]+boxes[...,2:3]/2
        boxes_y2 =boxes[...,1:2]+boxes[...,3:4]/2
    
    return boxes_x1,boxes_y1,boxes_x2,boxes_y2 # (nb_boxes , 1)

def calc_surface(boxes_x1:Tensor,boxes_y1:Tensor,
                 boxes_x2:Tensor,boxes_y2:Tensor)->Tensor:
    width ,height = boxes_x2-boxes_x1 ,  boxes_y2-boxes_y1

    return width.clamp(min=0) * height.clamp(min=0)    

    

def IoU(boxes_pred:Tensor,boxes_labels:Tensor,box_shape:str="midpoint"):
    """_summary_

    Args:
        boxes_pred (Tensor): Predictions of bounding boxes (Batch_size , 4)
        boxes_labels (Tensor): Ground truth bounding boxes (Batch_size , 4)
        box_shape (str, optional):  "midpoint" format or "corners" format

    Returns:
        tensor: IoU between each predicted bbx & the corresponding gt bbx
    """


    boxes_pred_x1,boxes_pred_y1,boxes_pred_x2,boxes_pred_y2 = \
        get_coords(boxes_pred,box_shape)
    boxes_labels_x1,boxes_labels_y1,boxes_labels_x2,boxes_labels_y2 = \
        get_coords(boxes_labels,box_shape)
  
    x_min = torch.max(boxes_pred_x1,boxes_labels_x1)
    x_max = torch.min(boxes_pred_x2,boxes_labels_x2)

    y_min = torch.max(boxes_pred_y1,boxes_labels_y1)
    y_max = torch.min(boxes_pred_y2,boxes_labels_y2)

    intersection = calc_surface(x_min,y_min,x_max,y_max)

    area_pred = calc_surface(boxes_pred_x1,boxes_pred_y1,boxes_pred_x2,boxes_pred_y2)
    area_label = calc_surface(boxes_labels_x1,boxes_labels_y1,boxes_labels_x2,boxes_labels_y2)

    union = area_pred + area_label - intersection

    return intersection / (union+1e-6)