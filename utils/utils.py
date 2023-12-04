import os
import sys
import json
import torch
from scipy.stats import truncnorm
from concurrent.futures import Future
from threading import Thread
from torchvision.transforms import ToPILImage
import ipdb
from utils.prepared_util import list2json
import numpy as np
from utils.evaluation import ACC_record, cor2IoU,cor2IoU2,mask2cor,maskfilter,msk2IoU_pre,mask2cor_rgb
from utils.IO import load_path_file
import math
PATH_FILE = load_path_file()
ROOT_PATH = PATH_FILE['root_path']

def compute_reg_acc(preds, targets, theta=0.5):
    # preds = box_transform_inv(preds.clone(), im_sizes)
    # preds = crop_boxes(preds, im_sizes)
    # targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    corr = (IoU >= theta).sum()
    return float(corr) / float(preds.size(0))

def compute_IoU(pred_box, gt_box):
    boxes1 = to_2d_tensor(pred_box)
    # boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes1[:, 2] = torch.clamp(boxes1[:, 0] + boxes1[:, 2], 0, 1)
    boxes1[:, 3] = torch.clamp(boxes1[:, 1] + boxes1[:, 3], 0, 1)

    boxes2 = to_2d_tensor(gt_box)
    boxes2[:, 2] = torch.clamp(boxes2[:, 0] + boxes2[:, 2], 0, 1)
    boxes2[:, 3] = torch.clamp(boxes2[:, 1] + boxes2[:, 3], 0, 1)
    # boxes2 = xywh_to_x1y1x2y2(boxes2)

    intersec = boxes1.clone()
    intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])

    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy

    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert ((a1 + a2 - ia < 0).sum() == 0)
    return ia / (a1 + a2 - ia)

def to_2d_tensor(inp):
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:
        inp = inp.unsqueeze(0)
    return inp


def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2 #由-1～1映射到0.1
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))

def to_gray_tensor(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = 38*tensor[0,:]+75*tensor[1,:]+15*tensor[2,:]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        raise Exception('The code here have bugs so i annotated it')
        #return torch.from_numpy(truncnorm.rvs(-truncation, truncation, size=size)).to(torch.float)


def save_common_run_params(args):
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')


def run_in_background(func: callable, *args, **kwargs):
    """ run f(*args, **kwargs) in background and return Future for its outputs """
    future = Future()

    def _run():
        try:
            future.set_result(func(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)

    Thread(target=_run).start()
    return future

class area_counter(object):
    '''bbox0:predict
    bbox1:real
    '''
    def __init__(self):
        self.area_list = np.array([0,0,0,0])
        self.length = np.array(0)
    
    def count(self,bbox0,bbox1):
        x0,y0,x1,y1 = bbox0
        x0_t,y0_t,x1_t,y1_t = bbox1
        ratio = ((x1-x0)*(y1-y0))/((x1_t-x0_t)*(y1_t-y0_t))
        self.length+=1
        if ratio<0.5:
            self.area_list[0]+=1
        elif 0.5<ratio and ratio<1:
            self.area_list[1]+=1
        elif 1<ratio and ratio<1.5:
            self.area_list[2]+=1
        else :
            self.area_list[3]+=1

    def __call__(self):
        return self.area_list/self.length




class acc_counter(object):
    def __init__(self):
        self.length = 0
        self.total = 0

    def reset(self):
        self.length = 0
        self.total = 0

    def __call__(self):
        return self.total/self.length
    
    def count(self,y_pred,y_tru):
        '''两个array变量
        '''
        self.total += (y_pred==y_tru).sum()
        self.length += len(y_pred)

class loss_counter(acc_counter):
    def __init__(self):
        super(loss_counter,self).__init__()

    def __call__(self):
        return super(loss_counter,self).__call__()
    
    def count(self,los):
        if torch.is_tensor(los):
            los=los.detach().item()
        self.total += los
        self.length += 1

class IoU_manager(object):
    '''inputs of __init__:
    thr_scale: list [], scheme of the filter threshold
    mark: indicate the way we generate bounding box, have three options:
    bbox_Psy, bbox, mask
    outputs:
    results of IoU of different threshold
    keys in acc_record
    '0.3'~'0.9';'miou','avg','box_v2'
    '''
    def __init__(self,thr_scale,mark='b2box',save_iou=False):
        self.thr_list = np.arange(thr_scale[0],thr_scale[1],thr_scale[2])
        acc_list = np.array([0 for x in self.thr_list])
        self.acc_record = ACC_record(acc_list,self.thr_list)
        self.mark = mark
        self.save_iou = save_iou
        if self.save_iou == True:
            self.iou_list=[]
    

    def bbox_update(self,box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        '''
        '''
        # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
            torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(
                b1_x1, b2_x1
            )  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = (
                    (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                    + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
                ) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif (
                    CIoU
                ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(
                        torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                    )
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou =  iou - (c_area - union) / c_area  # GIoU
        else:
            ioumax = [iou.max().item()]  # IoU
        self.acc_record.count(ioumax)
        return iou

    def update(self,mask_pre,gt):
        '''inputs:
        mask_pre: both (w,h) and (1,w,h) is acceptable
        gt: (w,h)
        outputs:
        cor_pre_list/mask_pre : predict location
        IoU_list : current img iou under different threshold
        mask2cor:
            输入:mask_pre:预测的mask(128*128),thr_list阈值列表[0.01,...,]
            输出:cor_pre_list:预测的mask列表[[x0,y0,x1,y1]...,[]] 范围都是0-128之间
        cor2IoU:
            输入:gt:[[tensor(x0),...,tensor(y1)]],一个列表,列表只有一个元素是一个tensor,
            tensor有四个数字组成,分别是x0,y0,x1,y1;cor_pre_list:同上
            输出:预测的mask和真实值之间的IoU
        '''
        mark = self.mark
        
        if mark == 'b2box':
            IoU_list = cor2IoU2(mask_pre,gt)
            self.acc_record.count(IoU_list)
            pre_list=mask_pre
        else:
            mask_pre = mask_pre.squeeze()
            if mark == 'bbox_psy':
                cor_pre_list = mask2cor_rgb(mask_pre,self.thr_list)
                IoU_list = cor2IoU2(cor_pre_list,gt)
                self.acc_record.count(IoU_list)
                pre_list=cor_pre_list

            elif mark == 'bbox':
                cor_pre_list = mask2cor(mask_pre,self.thr_list)
                IoU_list = cor2IoU(cor_pre_list,gt)
                self.acc_record.count(IoU_list)
                pre_list=cor_pre_list

            elif mark == 'avg_bbox':
                thr_list_tmp = float(mask_pre.mean())+self.thr_list
                cor_pre_list = mask2cor(mask_pre,thr_list_tmp)
                IoU_list = cor2IoU(cor_pre_list,gt)
                self.acc_record.count(IoU_list)
                pre_list=cor_pre_list

            elif mark =='mask':
                mask_pre = maskfilter(mask_pre.squeeze(),self.thr_list)
                IoU_list = msk2IoU_pre(mask_pre,gt)
                self.acc_record.count(IoU_list)
                pre_list = mask_pre
            else:
                raise Exception('undefined mark: {}'.format(mark))

        if self.save_iou == True:
            self.iou_list.append(IoU_list)
        return pre_list,IoU_list

        

    def acc_output(self,disp_form=True):
        '''elements in acc_output:
        {'GT-Loc':[self.acc_list_05], '0.4':[acc_04,acc_04_thr], '0.5':[acc_05,acc_05_thr], \
                '0.6':[acc_06,acc_06_thr], '0.7':[acc_07,acc_07_thr], '0.8':[acc_08,acc_08_thr], '0.9':[acc_09,acc_09_thr], \
                'avg':[avg],'miou':[miou,miou_thr],'box_v2':[box_v2]}
        '''
        return self.acc_record.acc_output(disp_form)

    def save_iou_list(self,id,dataset_name):
        iou_list = self.iou_list
        paths = os.path.join(ROOT_PATH,'results/IoU_data')
        file_name = '{}_{}_{}.json'.format(id,dataset_name,len(iou_list))

        save_path = os.path.join(paths,file_name)
        list2json(iou_list,save_path)
        return 0

def ReSize(bbox,size):
    w,h = size
    x0,y0,x1,y1 = bbox[0]
    return [[x0*w,y0*h,x1*w,y1*h]]


class Top_1_5(object):
    '''
    calculate top1/top5 list according to 
    clas results and iou_list 
    '''
    def __init__(self):
        self.IoU_list=[]
    
    def add(self,IoU_lis):
        self.IoU_list.append(IoU_lis)

    def save(self,paths):
        if len(self.IoU_list) == 0:
            raise Exception('the list is empty')
        else:
            list2json(self.IoU_list,paths)
    
    def reset(self):
        self.IoU_list = []

    def load(self,paths):
        if len(self.IoU_list) > 0:
            raise Exception('the list is not empty')
        else:
            with open(paths,'r') as f:
                self.IoU_list = json.load(f) 

    def pre_top1_top5(self,cls_paths):
        top1_list,top5_list = self._load_cls(cls_paths)
        IoU_list = self._select_iou_list(0.5)
        top1_loc = self._cal_cls_loc(IoU_list,top1_list)
        top5_loc = self._cal_cls_loc(IoU_list,top5_list)
        return top1_loc,top5_loc
    
    def _cal_cls_loc(self,IoU_list,top_list):
        ct = 0
        length = len(IoU_list)
        for i in range(length):
            if IoU_list[i]>0.5 and top_list[i]>50:
                ct+=1
        return ct/length


    def _select_iou_list(self,thr):
        IoU_list = np.array(self.IoU_list)
        acc_list = np.sum(IoU_list>0.5,axis=0)
        idx = np.argmax(acc_list)
        selected_iou = IoU_list[:,idx]
        return selected_iou


    def _load_cls(self,paths):
        cls1_path=os.path.join(paths,'data/classfication1.txt')
        cls5_path=os.path.join(paths,'data/classfication5.txt')
        with open(cls1_path,'r') as f:
            top1_list = json.load(f)
        with open(cls5_path,'r') as f:
            top5_list = json.load(f)
        return top1_list,top5_list