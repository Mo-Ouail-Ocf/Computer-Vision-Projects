import torch
from torch import Tensor
import torch.nn as nn
from utils import IoU

# nb boxes = 2 , nb cells = 7 , nb classes = 20
class YoloV1Loss(nn.Module):
    def __init__(self,lambda_noobj,lambda_coord):
        super().__init__()
        self.mse = nn.MSELoss(reduce='sum')
        self.lambda_noobj=lambda_noobj
        self.lambda_coord=lambda_coord

    def forward(self,preds:Tensor,targets:Tensor):
        # preds  : (batch_size , S , S , C + B *5 )  B = (x,y,w,h,c)
        # targets : (batch_size , S , S , C + Identity-cell-i (i+j) +(X,Y,W,H) )
        # loss = coords loss + dimensions loss + confidence loss + class proba loss
        
        identity_mask = targets[...,20:21] # (batch_size , S , S , 1 ) : exists object or not

        ###### 1- calc IoU for each bounding box and corresponding GT BBX #######
        #########################################################################
        gt_boxes = targets[...,21:25]
        # gt bbx for each grid cell (i,j) 
        
        boxes_pred_1 = preds[...,21:25] # first predicted bbx 
        iou_b1 = IoU(boxes_pred_1,gt_boxes) # (batch_size , s , s , 1)
        
        boxes_pred_2 = preds[...,26:30] # second predicted bbx
        iou_b2 = IoU(boxes_pred_2,gt_boxes) # (batch_size , s , s , 1)

        ###### 2- Get the predcited bbx responsible for pred for each grid cell #######
        ###############################################################################

        iou_combined = torch.cat([iou_b1,iou_b2],dim=-1) # (batch_size , s , s , 2)
        _ , best_bbox_indices  = torch.max(iou_combined,dim=-1,keepdim=True) # (batch_size , s , s , 1)

        ############################ 3-1 coord + dim loss ###############################
        ############################################################################
         
        # coord loss 

        # Extract the predicted (x,y,w,h) for each cell - > ( batch_size , s , s , 4)
        # we will extract them from preds using indexes we got wtih best_bbox_indices

        # here , I got the exact index in preds by Adjusting indices to x, y, w, h positions
        best_boxes_coord_indexes = (best_bbox_indices * 5) + torch.tensor([21, 22, 23, 24]).\
            unsqueeze(0).unsqueeze(0).unsqueeze(0) # ( batch_size , s , s , 4)
        
        predicted_coords = preds.gather(
            dim=-1 , index= best_boxes_coord_indexes
        ) # (batch_size , s ,s , 4)

        coord_loss = self.mse(identity_mask*predicted_coords[0:2],identity_mask*gt_boxes[0:2])

        target_w_h = torch.sqrt(gt_boxes[2:4])
        predicted_w_h = torch.sign(predicted_coords[2:4]) * torch.sqrt(torch.abs(predicted_coords[2:4])+1e-6)

        dim_loss = self.mse(predicted_w_h*identity_mask,target_w_h*identity_mask)


        ############################ 3-2 confidence loss ###########################
        ############################################################################