"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
from distutils.archive_util import make_archive
import os
import argparse
import random
import pickle
import sys
import ipdb
#将TokenCut路径加入path
sys.path.append( os.path.abspath(os.path.join(__file__,'../..')) )
#将gamma/main路径加入path
sys.path.append(os.getcwd())
# print(os.getcwd())
# ipdb.set_trace()
import torch
import datetime
import torch.nn as nn
import numpy as np
import json
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import normalize

from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou,ObjLocDataset
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
#os.getcwd()是返回命令行执行所在的路径，在本工程中是main
#一个经验是，import上一层级需要用绝对路径而不能用'..'这样的相对路径。
from utils.utils import IoU_manager
from utils.IO import PATH_FILE, Info_Record, Train_Record,Pretrained_Read
from utils.visulization import visualizer
from object_discovery import ncut,feat_extract,ncut2
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k",'CUB'],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test", "Sample"\
            ,"validation" , "train_list","val_list","test_list"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )
    parser.add_argument("--visualization", action="store_true", help="used for visulize eigenvector")

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v","F"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)
    # IO from Train.py
    parser.add_argument('--record_file', type=int, default=1)
    parser.add_argument('--similarity_matrix', type=str, default='Sample_12')
    args = parser.parse_args()
    #如果只是但张图片可视化，保存prediction没啥意义，有图片就行了
    #也不需要进行测试（因为可能是网图啊）
    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    param = args
    # Initialization Setting--------------------------------------------------------------------------
    info_record = Train_Record(file=param.record_file)
    info_record.record(__file__,'both')
    info_record.record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)
    
    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device('cuda') 
    model = get_model(args.arch, args.patch_size, device)

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.dataset != 'CUB':
        if args.image_path is not None:
            dataset = ImageDataset(args.image_path, args.resize)
        else:
            dataset = Dataset(args.dataset, args.set, args.no_hard)

        # -------------------------------------------------------------------------------------------------------
        # Directories
        if args.image_path is None:
            args.output_dir = os.path.join(args.output_dir, dataset.name)
        os.makedirs(args.output_dir, exist_ok=True)

        # Naming
        if args.dinoseg:
            # Experiment with the baseline DINO-seg
            if "vit" not in args.arch:
                raise ValueError("DINO-seg can only be applied to tranformer networks.")
            exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
        else:
            # Experiment with TokenCut 
            exp_name = f"TokenCut-{args.arch}"
            if "vit" in args.arch:
                exp_name += f"{args.patch_size}_{args.which_features}"

        print(f"Running TokenCut on the dataset {dataset.name} (exp: {exp_name})")

        # Visualization 
        if args.visualize:
            vis_folder = f"{args.output_dir}/{exp_name}"
            os.makedirs(vis_folder, exist_ok=True)
            
        if args.save_feat_dir is not None : 
            os.mkdir(args.save_feat_dir)

        # -------------------------------------------------------------------------------------------------------
        # Loop over images
        preds_dict = {}
        cnt = 0
        corloc = np.zeros(len(dataset.dataloader))
        
        start_time = time.time() 
        pbar = tqdm(dataset.dataloader)
    else:
    #---------------------------------------------------------------------------------------
    #Setting for CUB
        preds_dict = {}
        dataset = ObjLocDataset(args.dataset,args.set)
        cnt = 0
        corloc = np.zeros(len( dataset ))
        start_time = time.time() 
        pbar = tqdm(dataset)
        #name
        exp_name = f"TokenCut-{args.arch}"
        if "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

        print(f"Running TokenCut on the dataset {args.dataset} (exp: {exp_name})")
    
    # IoU manager initialization-----------------------------------------------
    #[0,0.01,0.1]是保证计算iou list的长度为1，具体数值无实际意义
    IoU_man = IoU_manager([0,0.01,0.1],mark='Box2IoU')
    vis = visualizer() 

    IoU_list = []
    Sim_list = []
    Name_list = []
    similarity_list= []
    sim_index_lis = []
    sim_score_lis = []
    loaded_path = os.path.join('/home/qxy/Desktop/gamma/results/similarity/',args.similarity_matrix+'.pth')
    similarity_matrix = torch.load(loaded_path).cuda()
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0].squeeze(0)
        init_image_size = img.shape
        if args.dataset!='CUB':
        # Get the name of the image
            im_name = dataset.get_image_name(inp[1])
            # Pass in case of no gt boxes in the image
            if im_name is None:
                continue
        else:
            im_name = im_id

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            if args.dataset != 'CUB':
                gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

                if gt_bbxs is not None:
                    # Discard images with no gt annotations
                    # Happens only in the case of VOC07 and VOC12
                    if gt_bbxs.shape[0] == 0 and args.no_hard:
                        continue
            else:
                gt_bbxs = np.asarray(inp[1])


        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit"  in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                #这里hook所引出的featout可以直接在的当前文件夹中引用
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                feats = []
                #elem 控制输出第elem层的特征
                for elem in range(11,12):
                    
                    hook = model._modules["blocks"][elem]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
                    # Forward pass in the model
                    attentions = model.get_last_selfattention(img[None, :, :, :])
                    #hook可以通过remove移除，如果不移除的话，后层的hook将会覆盖前层hook出现的结果
                    hook.remove()
                    # Scaling factor
                    scales = [args.patch_size, args.patch_size]
                    # Dimensions
                    nb_im = attentions.shape[0]  # Batch size
                    nh = attentions.shape[1]  # Number of heads
                    nb_tokens = attentions.shape[2]  # Number of tokens

                    # Baseline: compute DINO segmentatin technique proposed in the DINO paper
                    # and select the biggest component
                    if args.dinoseg:
                        pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                        pred = np.asarray(pred)
                    else:
                        # Extract the qkv features of the last attention layer
                        qkv = (
                            feat_out["qkv"]
                            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                            .permute(2, 0, 3, 1, 4)
                        )
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                        # Modality selection
                        if args.which_features == "k":
                            #feats = k[:, 1:, :]
                            #正常处理
                            feat = k
                            cls = feat[:,1,:]
                            similarity_list.append( cls.cpu())
                        elif args.which_features == "q":
                            #feats = q[:, 1:, :]
                            feats.append( q)
                        elif args.which_features == "v":
                            #feats = v[:, 1:, :]
                            feats.append( v)
                        elif args.which_features == "F" :
                            feat = k
                            cls = feat[:,1,:]
                            cos = torch.nn.CosineSimilarity(dim=1)
                            sim_score = cos(cls,similarity_matrix)
                            max_score,max_index = sim_score.topk(5)
                            min_score,min_index = (-sim_score).topk(5)
                            #top5 idx and last5 idx:
                            sim_index_lis.append([max_index,min_index])
                            sim_score_lis.append([max_score,min_score])

                        
                        if args.save_feat_dir is not None : 
                            np.save(os.path.join(args.save_feat_dir, im_name.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')), feats.cpu().numpy())
                            continue

            else:
                raise ValueError("Unknown model.")

            
        
        # ------------ Apply TokenCut ------------------------------------------- 
                        
        #IoU calculation -------------------------------------------------
        # Save the prediction

        
        # print(max(ious),im_id)

        # Evaluation
        # ------------ Visualizations -------------------------------------------
    np.set_printoptions(precision=4)
    end_time = time.time()
    if args.which_features == 'F' and args.visualization:
        save_paths = '{}'.format(info_record.id)
        save_paths = os.path.join('/home/qxy/Desktop/gamma/results/pictures',save_paths)
        os.mkdir(save_paths)
        #找到index对应的picture图片地址
        #图片复制到save_paths,命名以x-y，x为目标图片，y为相关排名第几的
        N = len(dataset.dataloader)
        for img_idx in tqdm(range(20)):
            for sim_idx in range(5):
                max_copyed_path = dataset.dataloader.get_image_paths( sim_index_lis[img_idx][0][sim_idx].item() )
                min_copyed_path = dataset.dataloader.get_image_paths( sim_index_lis[img_idx][1][sim_idx].item() )
                # img_id = '{}-{}-{}.jpg'.format(img_idx,'max',sim_idx)
                max_saved = os.path.join(save_paths,'{}-{}-{}.jpg'.format(img_idx,'max',sim_idx))
                min_saved = os.path.join(save_paths,'{}-{}-{}.jpg'.format(img_idx,'min',sim_idx))
                shutil.copy(max_copyed_path,max_saved)
                shutil.copy(min_copyed_path,min_saved)
            
            sim_score_lis[img_idx][0] = str(np.around(sim_score_lis[img_idx][0].cpu().numpy(),4))
            sim_score_lis[img_idx][1] = str(-1*np.around(sim_score_lis[img_idx][1].cpu().numpy(),4))

        score_paths = os.path.join(save_paths,'score.json')
        with open(score_paths,"w") as f:
            for elem in sim_score_lis:
                f.write(str(elem))
                f.write('\n')

        
        ipdb.set_trace()


        # for eigenvector in eigenvector_list:
        #     mask_pre = torch.tensor(eigenvector).unsqueeze(0)
        #     img = img.clone().detach()
        #     mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(img.shape[1],img.shape[2]),mode='bilinear',align_corners=True)
        #     # vis.save_htmp(mask_pre,bbox_gt= list(gt_bbxs),bbox_pre=list(pred)
        #     # ,image= img,paths=[save_paths])
            
        #     tensor_save = torch.tensor(eigenvector)
        #     paths = vis._gen_paths([save_paths])
        #     paths = paths.split('.png')[0]+'.pt'
        #     torch.save(tensor_save,paths)


        #     vis.save_htmp(mask_pre#,bbox_gt= list(gt_bbxs),bbox_pre=list(pred)
        #     ,image= img,paths=[save_paths])        


    if args.which_features == 'k':
        sim_save_path = '/home/qxy/Desktop/gamma/results/similarity/{}_saved.pth'.format(info_record.id)
        cated = torch.cat(similarity_list,dim=0)
        torch.save(cated,sim_save_path)
    '''

    eval_acc = IoU_man.acc_output(disp_form=True)
    info_record.record(eval_acc,'log_detail')
    print('0.5:',eval_acc['0.5'],';0.9:',eval_acc['0.9'],';mIoU:',eval_acc['miou'],';box_v2:',eval_acc['box_v2'])
    print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
    
    
    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)

    info_record.record('the program is processed over.')
    '''