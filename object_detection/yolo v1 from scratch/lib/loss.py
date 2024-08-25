import torch
from torch import Tensor
import torch.nn as nn
from utils import IoU
import torch.nn.functional as F
import typing as tt 
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

        """ 1- Get the object_mask / no_object_mask        """
        object_mask = self._get_identity_mask(targets)  # (batch_size, S, S, 1): 1 if object exists, else 0
        no_object_mask = 1-object_mask

        """ 2- Get best bounding boxes indexes (0 or 1 ) for each cell """
        pred_bbx_1 = preds[...,21:25]
        pred_bbx_2 = preds[...,26:30]
        target__bbxs = targets[...,21:25]

        iou1 = IoU(pred_bbx_1,target__bbxs)
        iou2 = IoU(pred_bbx_2,target__bbxs)

        best_bbxs,combined_iou = self._get_best_bbxs(iou1,iou2)

        """             3- Get predicted boxes & confidence for each grid cell """

        coord_preds ,confidence_preds = self._extract_pred_coords_confidences(preds,best_bbxs) #  (batch_size,s,s,4) . (batch_size,s,s,1) 

        """             4- Calculate the Localization loss : """
        coords_targets = targets[...,21:25]
        coord_loss = self.mse(coord_preds[0:2]*object_mask,coords_targets[0:2]*object_mask)

        w_h_preds  , w_h_targets = coord_preds[2:4], coords_targets[2:4]

        w_h_preds_sqrt = torch.sign(w_h_preds) * ( torch.sqrt(torch.abs(w_h_preds)+1e-6) )
        w_h_targets_sqrt = torch.sqrt(w_h_targets)
        dim_loss = self.mse(w_h_preds_sqrt*object_mask,w_h_targets_sqrt*object_mask)

        localization_loss = self.lambda_coord*(coord_loss+dim_loss)

        """             5- Calculate confidence loss            """
        confidence_targets = combined_iou.gather(-1,index=best_bbxs) # (batch_size,s,s,1)
        
        confidence_loss = self.mse(confidence_preds*object_mask,confidence_targets*object_mask)+\
                            self.mse(confidence_preds*no_object_mask,confidence_targets*no_object_mask)

        """ 6- Calculate classification loss  """
        
        predicted_logits = preds[...,0:20]
        predicted_probas = F.softmax(predicted_logits,dim=-1)
        target_probas = targets[...,0:20]

        classif_loss = nn.mse(predicted_probas*object_mask,target_probas*object_mask)

        total_loss = localization_loss+confidence_loss+classif_loss

        return total_loss

    def _get_identity_mask(self,targets:Tensor)->Tensor:
        return targets[...,20:21] # (batch_size, S, S, 1): 1 if object exists, else 0
    
    def _get_best_bbxs(self,iou1:Tensor,iou2:Tensor)->tt.Tuple[Tensor,Tensor]:
        combined_iou = torch.cat([iou1,iou2],dim=-1) # (batch_size,s,s,2)
        _,best_bbxs = torch.max(combined_iou,dim=-1,keepdim=True) # (batch_size,s,s,1)
        return best_bbxs , combined_iou# (batch_size,s,s,1) , (batch_size,s,s,2)
    
    def _extract_pred_coords_confidences(self,preds:Tensor,best_bbxs:Tensor)->tt.Tuple[Tensor]:
        # preds : (batch_size,s,s,C+B*5) , best_bbxs : (batch_size,s,s,1)
        pred_coords_confidences_indexes = best_bbxs*5 + torch.tensor([21,22,23,24,25]).\
            unsqueeze(0).unsqueeze(0).unsqueeze(0) #  (batch_size,s,s,5) 
        predictions = preds.gather(
            -1 , pred_coords_confidences_indexes
        )
        coord_preds = predictions[...,21:25] #  (batch_size,s,s,4)
        confidence_preds = predictions[...,25:26] #  (batch_size,s,s,1) 

        return coord_preds ,confidence_preds  #  (batch_size,s,s,4) . (batch_size,s,s,1) 
    



    
