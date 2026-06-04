import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import argparse
import warnings
warnings.filterwarnings('ignore')


def test(args):

    # initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='mevis_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'MeViS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = '../DB/RVOS/MeViS'
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_videos = set(data.keys())
    video_list = sorted([video for video in valid_videos])

    # inference
    for idx_, video in enumerate(video_list):
        print(idx_)
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # per-frame mask prediction
            ref_masks = []
            ref_scores = []
            ref_num = 10
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
                ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                ref_mask = (ref_mask > 0).float()
                ref_masks.append(ref_mask)

                # consider vision-text alignment in addition to segmentation confidence
                clip_text = alphaclip.tokenize([exp]).cuda()
                alpha = clip_preprocess_mask(ref_mask).cuda()
                image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                text_features = clip.encode_text(clip_text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                confidence = ref_score
                alignment = torch.matmul(image_features, text_features.transpose(0, 1))[0]
                
                if args.use_temporal_score:
                    # temporal_consistency
                    sims = []
                    for delta in [-2, -1, 1, 2]:
                        neighbor = i + delta
                        if 0 <= neighbor < video_len:
                            neighbor_img_features = clip.visual(imgs_clip[neighbor].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                            neighbor_img_features = neighbor_img_features / neighbor_img_features.norm(dim=-1, keepdim=True)
                            sim = torch.matmul(image_features, neighbor_img_features.transpose(0, 1))[0][0]
                            sims.append(sim.item())
                    
                    if len(sims) > 0:
                        temporal_consistency = sum(sims) / len(sims)
                    else:
                        temporal_consistency = 0.0
                    
                    score = 0.4 * confidence + 0.4 * alignment + 0.2 * temporal_consistency
                    print(f"Frame {i}: confidence={confidence.item():.4f}, alignment={alignment.item():.4f}, temporal_consistency={temporal_consistency:.4f}, score={score.item():.4f}")
                else:
                    score = confidence + alignment
                    print(f"Frame {i}: confidence={confidence.item():.4f}, alignment={alignment.item():.4f}, score={score.item():.4f}")
                
                ref_scores.append(score)

            # select reference frame with highest mask score
            best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)
            best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))
            # forward pass
            for i in range(best_i, video_len):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == video_len - 1:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

            # backward pass
            for i in range(best_i, -1, -1):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                # clear memory for each sequence
                if i == 0:
                    processor.clear_memory()

                # convert format
                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_temporal_score', action='store_true', help='Use temporal consistency for key frame selection')
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
