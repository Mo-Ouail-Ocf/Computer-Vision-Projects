import torch
from torch import Tensor
import torch.nn as nn
from utils import IoU

# nb boxes = 2 , nb cells = 7 , nb classes = 20

class YoloV1Loss(nn.Module):
    def __init__(self,lambda_noobj=0.5,lambda_coord=5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_noobj=lambda_noobj
        self.lambda_coord=lambda_coord

    def forward(self,preds:Tensor,targets:Tensor):
        # preds  : (batch_size , S , S , C + B *5 )  B = (c,x,y,w,h)
        # targets : (batch_size , S , S , C + Identity-cell-i (i+j) +(X,Y,W,H) )
        # loss = coords loss + dimensions loss + confidence loss + class proba loss

        """ 1- Get the object_mask """
        cell_object_mask = self._get_identity_mask(targets)  # (batch_size, S, S, 1): 1 if object exists, else 0

        """ 2- Get best bounding boxes indexes (0 or 1 ) for each cell """
        iou_b1 = IoU(preds[...,21:25],targets[...,21:25]) #(batch_size,s,s,1)
        iou_b2 = IoU(preds[...,26:30],targets[...,21:25])

        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0) # (2,batch_size, S, S, 1)

        _ , bestboxes = torch.max(ious,dim=0) # (batch_size, S, S, 1)
      
        """    3- Localization loss """
        box_predictions = cell_object_mask * (
            bestboxes*preds[...,26:30]+
            (1-bestboxes)*preds[...,21:25]
        )
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6))
        target_boxes = targets[...,21:25]*cell_object_mask
        target_boxes[...,2:4] = torch.sqrt(target_boxes[...,2:4])

        localization_loss = self.mse(box_predictions,target_boxes)

        """     4- Calculate the Confidence loss : """
        predicted_conf_obj = bestboxes*preds[...,25:26] + (1-bestboxes)*preds[...,20:21]

        loss_conf_obj = self.mse(
            predicted_conf_obj*cell_object_mask,
            targets[...,20:21]*cell_object_mask
        )

        loss_conf_no_obj = self.mse(
            preds[...,20:21]*(1-cell_object_mask),
            targets[...,20:21]*(1-cell_object_mask)
        ) + self.mse(
            preds[...,25:26]*(1-cell_object_mask),
            targets[...,20:21]*(1-cell_object_mask)
        )

        confidence_loss = loss_conf_obj+self.lambda_noobj*loss_conf_no_obj

        """     6- Calculate classification loss  """
        classif_loss = self.mse(
            cell_object_mask*preds[...,:20],
            cell_object_mask*targets[...,:20]
        )
        return self.lambda_coord*localization_loss+confidence_loss+classif_loss
        

    def _get_identity_mask(self,targets:Tensor)->Tensor:
        return targets[...,20:21] # (batch_size, S, S, 1): 1 if object exists, else 0
    
    


if __name__=="__main__":
    yolo_loss = YoloV1Loss()

    # Dummy input
    batch_size, S, C, B = 2, 7, 20, 2
    preds = torch.empty(batch_size, S, S, C + B * 5).fill_(3)  # (batch_size, S, S, C + B * 5)
    targets = torch.empty(batch_size, S, S, C + 5).fill_(2)    # (batch_size, S, S, C + 5)

    # Calculate loss
    loss = yolo_loss(preds, targets)


    print(loss)