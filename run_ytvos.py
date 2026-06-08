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
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth=args.alpha_clip_ckpt, device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='ytvos_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'Ref_YTVOS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = args.data_root
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    all_temporal_scores = []

    # inference
    selected_key_frames_logs = []
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
            
            # Resume logic: skip if all frames are already generated
            expected_files = [os.path.join(save_path, f + '.png') for f in frames]
            if os.path.exists(save_path) and all(os.path.exists(f) for f in expected_files):
                print(f"Skipping {video_name} - {exp_id} (already processed)")
                continue
                
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            mask_cache = {}

            # helper to compute scores for a given frame index
            def compute_frame_score(frame_idx, exp_text):
                words = tokenizer(exp_text, return_tensors='pt')['input_ids'].cuda()
                if frame_idx in mask_cache:
                    ref_mask, ref_score = mask_cache[frame_idx]
                else:
                    ref_mask, ref_score = evfsam.inference(imgs_sam[frame_idx].unsqueeze(0).cuda(), imgs_beit[frame_idx].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                    ref_mask = (ref_mask > 0).float()
                    mask_cache[frame_idx] = (ref_mask, ref_score)
                
                clip_text = alphaclip.tokenize([exp_text]).cuda()
                alpha = clip_preprocess_mask(ref_mask).cuda()
                image_features = clip.visual(imgs_clip[frame_idx].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                text_features = clip.encode_text(clip_text)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                conf = ref_score.item()
                align = torch.matmul(image_features, text_features.transpose(0, 1))[0].item()
                
                temp_consist = 0.0
                if args.use_temporal_score:
                    ious = []
                    for delta in [-2, -1, 1, 2]:
                        neighbor = frame_idx + delta
                        if 0 <= neighbor < video_len:
                            if neighbor in mask_cache:
                                neighbor_mask, _ = mask_cache[neighbor]
                            else:
                                n_mask, n_score = evfsam.inference(imgs_sam[neighbor].unsqueeze(0).cuda(), imgs_beit[neighbor].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                                n_mask = (n_mask > 0).float()
                                mask_cache[neighbor] = (n_mask, n_score)
                                neighbor_mask = n_mask
                            
                            intersection = (ref_mask * neighbor_mask).sum().item()
                            union = ((ref_mask + neighbor_mask) > 0).float().sum().item()
                            if union == 0:
                                iou = 1.0
                            else:
                                iou = intersection / union
                            ious.append(iou)
                    if len(ious) > 0:
                        temp_consist = sum(ious) / len(ious)
                
                score = (args.w_conf * conf + args.w_align * align + args.w_temp * temp_consist) if args.use_temporal_score else (conf + align)
                orig_score = conf + align
                
                if args.use_temporal_score:
                    all_temporal_scores.append(temp_consist)
                
                return ref_mask, conf, align, temp_consist, score, orig_score

            ref_num = args.num_references
            coarse_results = []
            
            # Stage 1: Coarse Search
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                ref_mask, conf, align, temp_consist, score, orig_score = compute_frame_score(i, exp)
                coarse_results.append({
                    'index': i, 'mask': ref_mask, 'conf': conf, 'align': align,
                    'temp': temp_consist, 'score': score, 'orig_score': orig_score
                })
                print(f"Coarse Frame {i}: conf={conf:.4f}, align={align:.4f}, temp={temp_consist:.4f}, score={score:.4f}, orig_score={orig_score:.4f}")
            
            # Identify coarse key frames
            original_best_result = max(coarse_results, key=lambda x: x['orig_score'])
            coarse_best_result = max(coarse_results, key=lambda x: x['score'])
            
            original_best_i = original_best_result['index']
            coarse_best_i = coarse_best_result['index']
            
            best_i = coarse_best_i
            best_ref_mask = coarse_best_result['mask']
            
            # Logging
            log_entry = {
                'video_name': video_name,
                'exp_id': exp_id,
                'frame_index': best_i,
                'confidence_score': coarse_best_result['conf'],
                'alignment_score': coarse_best_result['align'],
                'temporal_score': coarse_best_result['temp'],
                'final_score': coarse_best_result['score']
            }
            selected_key_frames_logs.append(log_entry)
            
            # Visual Debugging
            if idx_ < 20:
                debug_dir = os.path.join('debug', 'keyframe_comparison', f"{video_name}_{exp_id}")
                os.makedirs(debug_dir, exist_ok=True)
                
                import shutil
                orig_img_path = os.path.join(img_folder, video_name, frames[original_best_i] + '.jpg')
                coarse_img_path = os.path.join(img_folder, video_name, frames[coarse_best_i] + '.jpg')
                
                shutil.copy(orig_img_path, os.path.join(debug_dir, 'original_findtrack_keyframe.jpg'))
                shutil.copy(coarse_img_path, os.path.join(debug_dir, 'temporal_aware_keyframe.jpg'))

            # forward pass
            for i in range(best_i, video_len):
                if i == best_i:
                    mask_prob = processor.step(imgs_cutie[i].cuda(), best_ref_mask.squeeze(0), objects=[1])
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
                    mask_prob = processor.step(imgs_cutie[i].cuda(), best_ref_mask.squeeze(0), objects=[1])
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

    # Print average selected scores
    if len(selected_key_frames_logs) > 0:
        avg_conf = sum(log['confidence_score'] for log in selected_key_frames_logs) / len(selected_key_frames_logs)
        avg_align = sum(log['alignment_score'] for log in selected_key_frames_logs) / len(selected_key_frames_logs)
        avg_temp = sum(log['temporal_score'] for log in selected_key_frames_logs) / len(selected_key_frames_logs)
        
        print("\n" + "="*50)
        print("Selected Key Frames Statistics")
        print("="*50)
        print(f"Total key frames selected: {len(selected_key_frames_logs)}")
        print(f"Average selected confidence score: {avg_conf:.4f}")
        print(f"Average selected alignment score:  {avg_align:.4f}")
        print(f"Average selected temporal score:   {avg_temp:.4f}")
        print("="*50)

    if args.use_temporal_score and len(all_temporal_scores) > 0:
        import matplotlib.pyplot as plt
        
        mean_temp = np.mean(all_temporal_scores)
        std_temp = np.std(all_temporal_scores)
        min_temp = np.min(all_temporal_scores)
        max_temp = np.max(all_temporal_scores)
        cv_temp = std_temp / mean_temp if mean_temp != 0 else 0
        
        print("\n" + "="*50)
        print("Temporal Consistency Analysis Across ALL Candidates")
        print("="*50)
        print(f"Total candidate frames evaluated: {len(all_temporal_scores)}")
        print(f"Mean Temporal Score: {mean_temp:.4f}")
        print(f"Standard Deviation:  {std_temp:.4f}")
        print(f"Minimum Temporal Score:  {min_temp:.4f}")
        print(f"Maximum Temporal Score:  {max_temp:.4f}")
        print(f"Coefficient of Variation:  {cv_temp:.4f}")
        
        os.makedirs('debug', exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(all_temporal_scores, bins=50, color='skyblue', edgecolor='black')
        plt.title("Distribution of Temporal Consistency Scores Across All Candidate Frames")
        plt.xlabel("Temporal Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        hist_path = os.path.join('debug', 'temporal_score_histogram.png')
        plt.savefig(hist_path)
        print(f"Histogram saved to: {hist_path}")
        print("-" * 50)
        
        print("\nFinal Conclusion:")
        print("1. Is temporal consistency nearly constant?")
        if std_temp < 0.05:
            print("   -> Yes, the score is nearly constant across most frames due to very low standard deviation.")
        else:
            print("   -> No, there is noticeable variation across different frames.")
            
        print("2. Does temporal consistency provide meaningful discrimination between frames?")
        if cv_temp < 0.05:
            print("   -> No, the variation is too small relative to the mean to provide meaningful discrimination.")
        else:
            print("   -> Yes, the variation is sufficient to help distinguish stable frames from unstable ones.")
            
        print("3. Would temporal consistency significantly affect frame ranking?")
        if (std_temp * 0.2) < 0.02: 
            print("   -> No, given its weight (0.2), the variation is too small to significantly alter the ranking compared to confidence and alignment scores.")
        else:
            print("   -> Yes, the variance is high enough that when multiplied by its weight (0.2), it can change the top-ranked key frame.")
        print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_temporal_score', action='store_true')
    parser.add_argument('--data_root', default='../DB/RVOS/YTVOS')
    parser.add_argument('--alpha_clip_ckpt', default='weights/clip_l14_336_grit_20m_4xe.pth')
    parser.add_argument('--num_references', type=int, default=5)
    parser.add_argument('--tracker', default='cutie')
    parser.add_argument('--sam2_checkpoint', default=None)
    parser.add_argument('--sam2_config', default=None)
    parser.add_argument('--min_frame_distance', type=int, default=15)
    parser.add_argument('--multi_reference', action='store_true')
    parser.add_argument('--w_conf', type=float, default=0.5)
    parser.add_argument('--w_align', type=float, default=0.3)
    parser.add_argument('--w_temp', type=float, default=0.2)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
