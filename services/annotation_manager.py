"""
标注数据管理器
负责项目、图片、标注的增删改查
"""
import os
import json
from pathlib import Path
from datetime import datetime
import uuid
import threading
import yaml


class AnnotationManager:
    """标注数据管理器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.projects = {}
        self._lock = threading.Lock()
        self._load_all_projects()

    def _load_all_projects(self):
        """加载所有项目"""
        projects_file = self.data_dir / 'projects.json'
        if projects_file.exists():
            with open(projects_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.projects = {p['id']: p for p in data.get('projects', [])}

    def _save_all_projects(self):
        """保存所有项目"""
        projects_file = self.data_dir / 'projects.json'
        with open(projects_file, 'w', encoding='utf-8') as f:
            json.dump({
                'projects': list(self.projects.values()),
                'updated_at': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def create_project(self, project: dict) -> dict:
        """创建新项目"""
        with self._lock:
            # === 为主动学习新增默认字段 ===
            project['text_prompt'] = project.get('text_prompt', 'A polyp in a colonoscopy image')
            project['image_states'] = project.get('image_states', {})
            project['latest_batch'] = project.get('latest_batch', [])
            # ==============================

            project_dir = self.data_dir / project['id']
            project_dir.mkdir(exist_ok=True)
            
            (project_dir / 'images').mkdir(exist_ok=True)
            (project_dir / 'masks').mkdir(exist_ok=True)
            
            self.projects[project['id']] = project
            self._save_all_projects()
            
            # === 新增：自动生成 prompt.yaml ===
            self._generate_training_config(project['id'], project['text_prompt'])
            
            return project
    
    def set_latest_batch(self, project_id: str, batch_list: list):
        """记录当前轮次抽取的人工标注图片名单"""
        with self._lock:
            if project_id in self.projects:
                self.projects[project_id]['latest_batch'] = batch_list
                self._save_all_projects()

    def _generate_training_config(self, project_id: str, text_prompt: str):
        """根据 controller.py 模板自动生成 yaml 配置"""
        config_dir = self.data_dir / project_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "prompt.yaml"

        base_config = {
            "model": "sam3",
            "dataset": {
                "train_img_dir": f"data/{project_id}/images",
                "train_mask_dir": f"data/{project_id}/masks",
                "text_prompt": text_prompt
            },
            "train": {
                "epochs": 60,
                "batch_size": 5,
                "use_cluster": 0
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def update_image_state(self, project_id: str, image_name: str, new_state: str):
        """更新单张图片的状态"""
        with self._lock:
            if project_id in self.projects:
                proj = self.projects[project_id]
                if "image_states" not in proj:
                    proj["image_states"] = {}
                proj["image_states"][image_name] = new_state
                self._save_all_projects()

    def sync_image_states(self, project_id: str):
        """同步状态，将新图片设为未标注池 (unannotated)"""
        with self._lock:
            if project_id not in self.projects: return
            proj = self.projects[project_id]
            if "image_states" not in proj:
                proj["image_states"] = {}
            states = proj["image_states"]
            
            # 遍历当前项目的所有图片
            for img in proj.get('images', []):
                img_name = img['filename']
                if img_name not in states:
                    states[img_name] = "unannotated"
            self._save_all_projects()
    

    def get_project(self, project_id: str) -> dict:
        """获取项目"""
        return self.projects.get(project_id)

    def list_projects(self) -> list:
        """列出所有项目"""
        return list(self.projects.values())

    def update_project(self, project_id: str, updates: dict) -> dict:
        """更新项目"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            project.update(updates)
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()
            return project

    def delete_project(self, project_id: str):
        """删除项目"""
        with self._lock:
            if project_id in self.projects:
                del self.projects[project_id]
                self._save_all_projects()

                # 删除项目目录
                project_dir = self.data_dir / project_id
                if project_dir.exists():
                    import shutil
                    shutil.rmtree(project_dir)

    def update_project_images(self, project_id: str, images: list, image_dir: str):
        """更新项目图片列表"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            existing_annotations = {}
            existing_status = {} # 新增：保留已有状态

            for img in project.get('images', []):
                if img.get('annotations'):
                    existing_annotations[img['filename']] = img['annotations']
                if 'status' in img:
                    existing_status[img['filename']] = img['status']

            for img in images:
                filename = img['filename']
                # 新增状态机属性
                img['status'] = existing_status.get(filename, 'unannotated') # unannotated, pending_hard, human_labeled, auto_solved
                img['unc_score'] = 0.0 
                
                if filename in existing_annotations:
                    img['annotations'] = existing_annotations[filename]
                    img['annotated'] = True

            project['images'] = images
            project['image_dir'] = image_dir
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()
            self._save_project_annotations(project_id)

    def add_annotations(self, project_id: str, image_index: int,
                        annotations: list, label: str = None):
        """添加标注（来自SAM3分割结果）"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]

            # 为每个标注添加类别标签
            for ann in annotations:
                if label:
                    ann['class_name'] = label
                if 'id' not in ann:
                    ann['id'] = str(uuid.uuid4())[:8]

            # 追加到现有标注
            if 'annotations' not in image:
                image['annotations'] = []
            image['annotations'].extend(annotations)
            image['annotated'] = True

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def save_annotations(self, project_id: str, image_index: int, annotations: list):
        """保存标注（覆盖）"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]
            image['annotations'] = annotations
            image['annotated'] = len(annotations) > 0

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def get_annotations(self, project_id: str, image_index: int) -> list:
        """获取标注"""
        project = self.projects.get(project_id)
        if not project:
            return []

        images = project.get('images', [])
        if image_index >= len(images):
            return []

        return images[image_index].get('annotations', [])

    def update_annotation(self, project_id: str, image_index: int,
                          annotation_id: str, updates: dict):
        """更新单个标注"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]
            annotations = image.get('annotations', [])

            for ann in annotations:
                if ann.get('id') == annotation_id:
                    ann.update(updates)
                    break

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def delete_annotation(self, project_id: str, image_index: int, annotation_id: str):
        """删除单个标注"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            image = project['images'][image_index]
            annotations = image.get('annotations', [])

            image['annotations'] = [
                ann for ann in annotations if ann.get('id') != annotation_id
            ]
            image['annotated'] = len(image['annotations']) > 0

            project['updated_at'] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def update_classes(self, project_id: str, classes: list):
        """更新类别列表"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            project['classes'] = classes
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()

    def _save_project_annotations(self, project_id: str):
        """保存项目标注到单独文件"""
        project = self.projects.get(project_id)
        if not project:
            return

        project_dir = self.data_dir / project_id
        project_dir.mkdir(exist_ok=True)

        annotations_file = project_dir / 'annotations.json'
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump({
                'project_id': project_id,
                'images': project.get('images', []),
                'classes': project.get('classes', []),
                'updated_at': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def mark_image_annotated(self, project_id: str, image_index: int, annotated: bool = True):
        """标记图片为已标注/未标注"""
        with self._lock:
            if project_id not in self.projects:
                raise ValueError(f"项目不存在: {project_id}")

            project = self.projects[project_id]
            if image_index >= len(project['images']):
                raise ValueError(f"图片索引越界: {image_index}")

            project['images'][image_index]['annotated'] = annotated
            project['updated_at'] = datetime.now().isoformat()

            self._save_all_projects()

    def get_annotation_stats(self, project_id: str) -> dict:
        """获取标注统计"""
        project = self.projects.get(project_id)
        if not project:
            return {}

        images = project.get('images', [])
        total = len(images)
        annotated = sum(1 for img in images if img.get('annotated', False))
        total_annotations = sum(
            len(img.get('annotations', [])) for img in images
        )

        return {
            'total_images': total,
            'annotated_images': annotated,
            'unannotated_images': total - annotated,
            'total_annotations': total_annotations,
            'progress': annotated / total * 100 if total > 0 else 0
        }
    
    def update_image_status(self, project_id: str, image_index: int, status: str, unc_score: float = 0.0):
        """更新单张图片的主动学习状态"""
        with self._lock:
            project = self.projects.get(project_id)
            if project and 0 <= image_index < len(project['images']):
                project['images'][image_index]['status'] = status
                project['images'][image_index]['unc_score'] = unc_score
                self._save_all_projects()
                self._save_project_annotations(project_id)

    def get_images_by_status(self, project_id: str, status: str) -> list:
        """根据状态获取图片索引和信息"""
        project = self.projects.get(project_id)
        if not project: return []
        return [(i, img) for i, img in enumerate(project['images']) if img.get('status', 'unannotated') == status]

    def batch_update_auto_solved(self, project_id: str, solved_data: list):
        """批量更新被机器自动解决的样本"""
        with self._lock:
            project = self.projects.get(project_id)
            if not project: return
            
            for item in solved_data:
                idx = item['index']
                project['images'][idx]['status'] = 'auto_solved'
                project['images'][idx]['unc_score'] = item['unc_score']
                project['images'][idx]['annotations'] = item['annotations']
                project['images'][idx]['annotated'] = True
                
            self._save_all_projects()
            self._save_project_annotations(project_id)
