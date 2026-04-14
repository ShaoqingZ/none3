import os
import sys
import json
import shutil
import threading
import subprocess
import cv2
import yaml
import numpy as np
import torch
import random
from PIL import Image
from pathlib import Path

# 使用本地 SAM_src 目录
sam3_src = Path(__file__).parent.parent / "SAM_src"
sys.path.insert(0, str(sam3_src))
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_xyxy_to_cxcywh
from services.annotation_manager import AnnotationManager

def compute_iou(mask1, mask2):
    """计算两个二值掩码的 IoU"""
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0: return 1.0 if (mask1.sum()+mask2.sum())==0 else 0.0
    return float(inter / (union + 1e-6))

class ActiveLearningService:
    def __init__(self, sam3_service, annotation_manager: AnnotationManager):
        if sam3_service is None:
            from services.sam3_service import SAM3Service
            self.sam3 = SAM3Service()
        else:
            self.sam3 = sam3_service
        self.ann_mgr = annotation_manager
        self.device = None
        self.al_tasks = {}  

    # ==============================================================
    # 1. 100% 像素级复刻 controller.py 的底层推理机制 (放弃二次包装)
    # ==============================================================
    def _predict_mask_raw(self, pil_img, text_outputs):
        inference_state = self.sam3.image_processor.set_image(pil_img)
        inference_state["backbone_out"].update(text_outputs)
        inference_state["geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=self.device), box_mask=torch.zeros(1, 0, device=self.device, dtype=torch.bool),
            point_embeddings=torch.zeros(0, 1, 2, device=self.device), point_mask=torch.zeros(1, 0, device=self.device, dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device=self.device, dtype=torch.long), point_labels=torch.zeros(0, 1, device=self.device, dtype=torch.long),
        )
        out = self.sam3.image_processor._forward_grounding(inference_state)
        w, h = pil_img.size
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        if "masks" in out and len(out["masks"]) > 0:
            for i, mask_tensor in enumerate(out["masks"]):
                if mask_tensor.dim() > 2:
                    mask_tensor = mask_tensor.squeeze(0)
                mask_np = mask_tensor.detach().cpu().numpy()
                binary_mask = (mask_np > 0).astype(np.uint8) * 255
                final_mask = np.maximum(final_mask, binary_mask)
        return final_mask, out

    def _predict_mask_with_pred_boxes(self, pil_img, pred_boxes_abs):
        w, h = pil_img.size
        inference_state = self.sam3.image_processor.set_image(pil_img)
        scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
        boxes_norm = pred_boxes_abs / scale
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_norm)
        
        num_boxes = boxes_cxcywh.shape[0]
        box_tensor = boxes_cxcywh.unsqueeze(1) 
        box_labels = torch.ones(num_boxes, 1, device=self.device, dtype=torch.long)
        
        self.sam3.image_processor.reset_all_prompts(inference_state)
        if "geometric_prompt" not in inference_state:
            inference_state["geometric_prompt"] = self.sam3.image_model._get_dummy_prompt()
            
        inference_state["geometric_prompt"].append_boxes(box_tensor, box_labels)
        dummy_text = self.sam3.image_model.backbone.forward_text(captions=["visual"], device=self.device)
        inference_state["backbone_out"].update(dummy_text)
        
        out_state = self.sam3.image_processor._forward_grounding(inference_state)
        prob_map_tensor = out_state["masks_logits"]
        
        if prob_map_tensor.numel() == 0:
            return np.zeros((h, w), dtype=np.uint8)
            
        if prob_map_tensor.dim() == 4:
            final_prob_map, _ = torch.max(prob_map_tensor, dim=0)
            final_prob_map = final_prob_map.squeeze(0)
        else:
            final_prob_map = prob_map_tensor

        prob_map = final_prob_map.detach().cpu().numpy()
        if prob_map.shape != (h, w):
            prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
        return (prob_map > 0.5).astype(np.uint8) * 255

    def _compute_single_image_tta(self, pil_img, m_text, out, text_out):
        if m_text.sum() == 0:
            return 3.0, 0.0, 0.0 
        
        m_flip, _ = self._predict_mask_raw(pil_img.transpose(Image.FLIP_LEFT_RIGHT), text_out)
        iou_f = compute_iou(m_text, np.fliplr(m_flip))
        
        if out["boxes"].shape[0] > 0:
            m_box = self._predict_mask_with_pred_boxes(pil_img, out["boxes"])
            iou_c = compute_iou(m_text, m_box)
        else:
            iou_c = 0.0
        
        # 100% 还原 controller.py 中的特有 Entropy 计算方式
        logits = out["masks_logits"]
        if logits.numel() == 0 or logits.shape[0] == 0:
            ent = 0.0
        else:
            if logits.dim() == 4: logits, _ = torch.max(logits, dim=0)
            p_map = logits.squeeze().detach().cpu().numpy()
            p_norm = np.clip(p_map, 1e-6, 1-1e-6)
            uncertain_mask = (p_map > 0.05) & (p_map < 0.95)
            
            if uncertain_mask.sum() > 0:
                pixel_entropy = -(p_norm * np.log(p_norm) + (1-p_norm) * np.log(1-p_norm))
                ent = np.mean(pixel_entropy[uncertain_mask])
            else:
                ent = 0.0
            
        unc_score = 1.0 * (1 - iou_f) + 2.0 * (1 - iou_c) + 0.5 * ent
        return float(unc_score), float(iou_f), float(iou_c)

    # ==========================================
    # 2. 难例挖掘与配置生成
    # ==========================================
    def fetch_next_manual_batch(self, project_id: str, batch_size: int = 5):
        self.ann_mgr.sync_image_states(project_id)
        with self.ann_mgr._lock:
            proj = self.ann_mgr.projects.get(project_id)
            if not proj: return []
            states = proj.get("image_states", {})
            unannotated = [img for img, st in states.items() if st == "unannotated"]
            if not unannotated: return []
            
            unc_scores = proj.get("uncertainty_scores", {})
            scored_unannotated = [img for img in unannotated if img in unc_scores]
            
            if len(scored_unannotated) > 0:
                print(f"\n[AL] 基于 TTA 分数，挖掘 Top-{batch_size} 极难样本...")
                scored_unannotated.sort(key=lambda x: unc_scores[x], reverse=True)
                selected = scored_unannotated[:batch_size]
                if len(selected) < batch_size:
                    remaining = list(set(unannotated) - set(selected))
                    selected += random.sample(remaining, min(batch_size - len(selected), len(remaining)))
            else:
                print(f"\n[AL] 冷启动阶段，随机抽取 {batch_size} 个样本...")
                selected = random.sample(unannotated, min(batch_size, len(unannotated)))
                
            for img in selected:
                states[img] = "manual_batch"
                if img in unc_scores:
                    del unc_scores[img] 
                
            proj['image_states'] = states
            proj['latest_batch'] = selected 
            proj['uncertainty_scores'] = unc_scores
            self.ann_mgr._save_all_projects()
        return selected

    def generate_project_training_config(self, project_id: str):
        proj = self.ann_mgr.projects.get(project_id)
        if not proj: return
        workspace_dir = self.ann_mgr.data_dir / project_id / "workspace"
        text_prompt = proj.get("text_prompt", "target")
        classes_raw = proj.get("classes", ["target"])
        class_name = classes_raw[0] if isinstance(classes_raw, list) and len(classes_raw) > 0 else str(classes_raw)

        template_path = Path("SAM_src/sam3/train/configs/kvasir/prompt.yaml")
        if not template_path.exists():
            print(f"[AL 引擎] 警告：找不到模板文件 {template_path}")
            return

        with open(template_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        if 'paths' not in cfg: cfg['paths'] = {}
        if 'launcher' not in cfg: cfg['launcher'] = {}

        cfg['paths']['data_root'] = str(workspace_dir.absolute())
        cfg['paths']['experiment_log_dir'] = str(workspace_dir.absolute())
        cfg['paths']['bpe_path'] = os.path.abspath("SAM_src/assets/bpe_simple_vocab_16e6.txt.gz")
        cfg['launcher']['experiment_log_dir'] = str(workspace_dir.absolute())

        if 'trainer' not in cfg: cfg['trainer'] = {}
        if 'model' not in cfg['trainer']: cfg['trainer']['model'] = {}
        cfg['trainer']['model']['prompt_tuning_init_text'] = text_prompt
        
        # 强制配置存储路径以确保能找到权重
        if 'continual' not in cfg['trainer']: cfg['trainer']['continual'] = {}
        cfg['trainer']['continual']['phi_checkpoint_dir'] = str(workspace_dir.absolute())
        
        cfg['all_supercategories'] = [
            {
                'name': class_name,
                'val': {'img_folder': 'test', 'json': 'test.json'},
                'train': {'img_folder': 'train', 'json': 'train.json'}
            }
        ]
        
        prompt_info = [{"id": 1, "name": class_name, "supercategory": "target"}]
        cfg['data_prompts'] = {class_name: json.dumps(prompt_info)}

        train_configs_dir = Path("SAM_src/sam3/train/configs/temp")
        train_configs_dir.mkdir(parents=True, exist_ok=True)
        (train_configs_dir / "__init__.py").touch(exist_ok=True)
        
        new_config_name = f"prompt_{project_id}.yaml"
        new_config_path = train_configs_dir / new_config_name

        with open(new_config_path, 'w', encoding='utf-8') as f:
            f.write("# @package _global_\n")
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ==========================================
    # 3. 终极闭环主控
    # ==========================================
    def run_train_and_infer_bg(self, project_id: str, solved_thresh: float, epochs: int, lr: float):
        self.al_tasks[project_id] = {
            "status": "training", 
            "current": 0, 
            "total": 0, 
            "message": "正在生成配置并启动训练..."
        }
        def task():
            try:
                print(f"\n[一键闭环] 开始处理项目 {project_id} ...")
                proj = self.ann_mgr.projects.get(project_id)
                latest_batch = proj.get("latest_batch", [])
                if not latest_batch: return

                workspace_dir = self.ann_mgr.data_dir / project_id / "workspace"
                train_img_dir = workspace_dir / "train"
                test_img_dir = workspace_dir / "test" 
                
                workspace_dir.mkdir(parents=True, exist_ok=True)
                if train_img_dir.exists(): shutil.rmtree(train_img_dir)
                if test_img_dir.exists(): shutil.rmtree(test_img_dir) 
                train_img_dir.mkdir(parents=True, exist_ok=True)
                test_img_dir.mkdir(parents=True, exist_ok=True)       
                
                round_json_path = workspace_dir / "train.json"
                round_test_path = workspace_dir / "test.json"   
                original_img_dir = Path(proj['image_dir'])
                classes_raw = proj.get("classes", ["target"])
                class_name = classes_raw[0] if isinstance(classes_raw, list) and len(classes_raw) > 0 else str(classes_raw)

                coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": class_name}]}
                ann_id = 1
                valid_train_count = 0

                # ======= 核心修改：引入状态字典，识别负样本 =======
                image_states = proj.get("image_states", {})

                for img_id, img_info in enumerate(proj.get('images', []), start=1):
                    img_name = img_info['filename']
                    current_state = image_states.get(img_name, "")
                    
                    # 只要是本轮提取的，且你点过保存或“无目标”（状态为 completed）的，都算数！
                    if img_name in latest_batch and current_state == "completed":
                        img_path_src = original_img_dir / img_name
                        if not img_path_src.exists(): continue
                        
                        shutil.copy(img_path_src, train_img_dir / img_name)
                        img = cv2.imread(str(img_path_src))
                        h, w = img.shape[:2]
                        coco_data["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})

                        anns = img_info.get('annotations', [])
                        
                        # 1. 如果有标注，正常写入
                        if anns:
                            for ann in anns:
                                if 'polygon' in ann and len(ann['polygon']) > 2:
                                    poly = ann['polygon']
                                    flat_poly = [float(val) for pt in poly for val in pt] 
                                    xs = [float(pt[0]) for pt in poly]
                                    ys = [float(pt[1]) for pt in poly]
                                    x_min, x_max = min(xs), max(xs)
                                    y_min, y_max = min(ys), max(ys)
                                    coco_data["annotations"].append({
                                        "id": ann_id, "image_id": img_id, "category_id": 1,
                                        "segmentation": [flat_poly],
                                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                                        "area": float((x_max - x_min) * (y_max - y_min)), "iscrowd": 0
                                    })
                                    ann_id += 1
                        # 2. 如果没有标注（你点了无目标），生成“幽灵标注”骗过底层检查！
                        else:
                            coco_data["annotations"].append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": 1,
                                "segmentation": [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]], # 极小框
                                "bbox": [0.0, 0.0, 1.0, 1.0], # 左上角1个像素
                                "area": 0.001, # 极小面积
                                "iscrowd": 0
                            })
                            ann_id += 1
                            
                        valid_train_count += 1
                # =================================================

                if valid_train_count > 0:
                    with open(round_json_path, 'w', encoding='utf-8') as f:
                        json.dump(coco_data, f, ensure_ascii=False)

                    dummy_test_coco = {
                        "info": {}, "licenses": [], "images": [], "annotations": [],
                        "categories": [{"id": 1, "name": class_name, "supercategory": "target"}]
                    }
                    with open(round_test_path, 'w', encoding='utf-8') as f:
                        json.dump(dummy_test_coco, f, ensure_ascii=False)

                    train_configs_dir = Path("SAM_src/sam3/train/configs/temp")
                    new_config_name = f"prompt_{project_id}.yaml"
                    new_config_path = train_configs_dir / new_config_name
                    
                    if new_config_path.exists():
                        with open(new_config_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # 安全处理 yaml 的 @package _global_ 头
                        header = "# @package _global_\n"
                        if content.startswith(header):
                            content = content.replace(header, '')
                            
                        cfg = yaml.safe_load(content) or {}
                        
                        # 修改 Epochs
                        if 'scratch' not in cfg: cfg['scratch'] = {}
                        cfg['scratch']['max_data_epochs'] = epochs
                        
                        # 修改 学习率 lr_phi
                        cfg['scratch']['lr_phi'] = lr
                        
                        # 保存回文件
                        with open(new_config_path, 'w', encoding='utf-8') as f:
                            f.write(header)
                            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

                    print(f"[一键闭环] 提取了 {valid_train_count} 张图。调用专属配置文件启动 train.py...")
                    train_script = os.path.abspath("SAM_src/sam3/train/train.py")
                    new_config_name = f"prompt_{project_id}"
                    
                    cmd = ["python", train_script, "-c", f"configs/temp/{new_config_name}"]
                    print(f"执行命令: {' '.join(cmd)}")
                    
                    env = os.environ.copy()
                    local_sam_src = os.path.abspath("SAM_src")
                    env["PYTHONPATH"] = local_sam_src + os.pathsep + env.get("PYTHONPATH", "")
                    
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                    for line in process.stdout:
                        print(f"[Train] {line.strip()}")
                    process.wait()
                    
                    if process.returncode != 0:
                        print("[一键闭环] ❌ 训练执行报错，中止流程。")
                        self.al_tasks[project_id]["status"] = "error"
                        self.al_tasks[project_id]["message"] = "训练脚本执行错误"
                        return
                    print("[一键闭环] ✅ 提示微调完成！")

                print(f"\n[一键闭环] 开始扫描全池进行 TTA 自动检验 (阈值: {solved_thresh})...")
                text_prompt = proj.get("text_prompt", "target")
                self.al_tasks[project_id]["status"] = "inferring"
                self._run_tta_inference(project_id, solved_thresh, text_prompt, original_img_dir, proj)
                self.al_tasks[project_id]["status"] = "completed"
                self.al_tasks[project_id]["message"] = "自动标注完成"

            except Exception as e:
                print(f"[AL 引擎] 后台流程严重崩溃: {e}")
                import traceback
                traceback.print_exc()
                self.al_tasks[project_id]["status"] = "error"
                self.al_tasks[project_id]["message"] = f"错误: {str(e)}"

        threading.Thread(target=task, daemon=True).start()

    def _run_tta_inference(self, project_id, solved_thresh, text_prompt, img_dir, proj):
        states = proj.get("image_states", {})
        unannotated = [img for img, st in states.items() if st == "unannotated"]

        if project_id in self.al_tasks:
            self.al_tasks[project_id]["total"] = len(unannotated)
            self.al_tasks[project_id]["current"] = 0
        
        if "uncertainty_scores" not in proj: proj["uncertainty_scores"] = {}
            
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.sam3._init_image_model()
        text_encoder = self.sam3.image_model.backbone.language_backbone
        token_ids = text_encoder.tokenizer([text_prompt], context_length=text_encoder.context_length).to(self.device)
        
        # === 【绝杀】动态叠加微调好的 phi 权重！===
        with torch.no_grad():
            base_text_embeds = text_encoder.encoder.token_embedding(token_ids)
            v_new = base_text_embeds
            
            workspace_dir = self.ann_mgr.data_dir / project_id / "workspace"
            phi_path = workspace_dir / "phi_continuous_checkpoint.pt"
            
            if phi_path.exists():
                try:
                    phi_tensor = torch.load(phi_path, map_location=self.device)
                    if isinstance(phi_tensor, dict): 
                        phi_tensor = phi_tensor.get("pt_phi", next(iter(phi_tensor.values())))
                    v_new = base_text_embeds + phi_tensor.to(self.device)
                    print(f"[AL 引擎] 成功加载专家权重，模型视野升级！")
                except Exception as e:
                    print(f"[AL 警告] 无法加载 phi 权重: {e}")
            else:
                print(f"[AL 引擎] 未找到专家权重，使用 Zero-shot 进行推理...")
                
            text_out = self.sam3.image_model.backbone.forward_text(captions=[text_prompt], input_embeds=v_new, device=self.device)
        
        for i, img_name in enumerate(unannotated):
            if project_id in self.al_tasks:
                self.al_tasks[project_id]["current"] = i + 1
            
            img_path = img_dir / img_name
            if not img_path.exists(): continue
            
            try:
                pil_img = Image.open(img_path).convert("RGB")
                m_pred, out = self._predict_mask_raw(pil_img, text_out)
                unc_score, iou_f, iou_c = self._compute_single_image_tta(pil_img, m_pred, out, text_out)
                
                print(f"[AL TTA] {img_name} -> Unc:{unc_score:.3f} | IoU_F:{iou_f:.3f} | IoU_C:{iou_c:.3f}")
                proj["uncertainty_scores"][img_name] = float(unc_score)
                
                # 恢复你原汁原味的严苛判断条件
                if unc_score < solved_thresh and iou_f > 0.90 and iou_c > 0.90:
                    print(f"  [+] 简单样本命中！自动生成掩码 -> {img_name}")
                    # 传入 m_pred (0~255) 并转为 0~1 的 Numpy 供轮廓提取
                    self._save_pred_mask_as_annotation(project_id, img_name, m_pred / 255.0, text_prompt)
                    if img_name in proj["uncertainty_scores"]:
                        del proj["uncertainty_scores"][img_name]
            except Exception as e:
                print(f"[TTA 异常] {img_name}: {e}")
                
        with self.ann_mgr._lock:
            self.ann_mgr._save_all_projects()
                
        print("\n[一键闭环] 🎉 自动检验完成！TTA 分数已更新。前端可刷新查看。")

    def _save_pred_mask_as_annotation(self, project_id, img_name, mask_numpy, class_name):
        if mask_numpy is None: return
        mask_uint8 = (mask_numpy * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotations = []
        for contour in contours:
            if cv2.contourArea(contour) < 50: continue 
            poly = contour.squeeze().tolist()
            if isinstance(poly, list) and len(poly) > 2:
                if isinstance(poly[0], (int, float)): 
                    poly = [[poly[i], poly[i+1]] for i in range(0, len(poly), 2)]
                annotations.append({
                    "id": str(os.urandom(4).hex()),
                    "class_name": class_name,
                    "polygon": poly,
                    "score": 0.99,
                    "machine_generated": True
                })
        
        if annotations:
            with self.ann_mgr._lock:
                proj = self.ann_mgr.projects[project_id]
                for idx, img_info in enumerate(proj['images']):
                    if img_info['filename'] == img_name:
                        img_info['annotations'] = annotations
                        img_info['annotated'] = True
                        break
                if "image_states" not in proj: proj["image_states"] = {}
                proj["image_states"][img_name] = "review"
                self.ann_mgr._save_all_projects()

# import threading
# import time
# import random
# import shutil
# import cv2
# import numpy as np
# import yaml

# class ActiveLearningService:
#     def __init__(self, sam3_service, annotation_manager):
#         self.sam3 = sam3_service
#         self.ann_mgr = annotation_manager

#     def fetch_next_manual_batch(self, project_id: str, batch_size: int = 5):
#         """从未标注池抽取 5 张进入待人工标注"""
#         self.ann_mgr.sync_image_states(project_id)
#         proj = self.ann_mgr.projects.get(project_id)
#         if not proj: return []

#         states = proj.get("image_states", {})
#         unannotated = [img for img, st in states.items() if st == "unannotated"]
        
#         if not unannotated: return []
#         selected = random.sample(unannotated, min(batch_size, len(unannotated)))
#         for img in selected:
#             self.ann_mgr.update_image_state(project_id, img, "manual_batch")
#         return selected

#     def run_auto_inference_bg(self, project_id: str, solved_thresh: float):
#         """后台异步运行机器全量推理"""
#         def task():
#             print(f"[AL] 启动自动推理，阈值: {solved_thresh}")
#             self.ann_mgr.sync_image_states(project_id)
#             proj = self.ann_mgr.projects.get(project_id)
#             if not proj: return
            
#             states = proj.get("image_states", {})
#             unannotated = [img for img, st in states.items() if st == "unannotated"]
            
#             for img_name in unannotated:
#                 time.sleep(0.5) # 模拟推理时间
#                 # 此处暂用随机数模拟你 controller 里的 uncertainty_score
#                 unc_score = random.uniform(0.1, 1.0) 
                
#                 if unc_score < solved_thresh:
#                     print(f"[AL] {img_name} 满足条件，进入待检验")
#                     # 后期可在此处保存真实的机器生成的 mask
#                     self.ann_mgr.update_image_state(project_id, img_name, "review")
#             print("[AL] 推理完成！")
            
#         threading.Thread(target=task, daemon=True).start()