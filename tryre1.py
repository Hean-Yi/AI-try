import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging


# ============================================================================
# 1. 数据增强策略
# ============================================================================

class WeakAugmentation:
    """弱增强策略"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __call__(self, x):
        return self.transform(x)


class StrongAugmentation:
    """强增强策略"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __call__(self, x):
        return self.transform(x)


# ============================================================================
# 2. 双视图数据集包装器
# ============================================================================

class DualViewDataset(Dataset):
    """双视图数据集包装器，返回弱增强和强增强视图"""
    
    def __init__(self, dataset: Dataset, weak_aug=None, strong_aug=None, 
                 return_index: bool = False):
        self.dataset = dataset
        self.weak_aug = weak_aug or WeakAugmentation()
        self.strong_aug = strong_aug or StrongAugmentation()
        self.return_index = return_index
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset, 'indices'):
            # 处理Subset情况
            original_idx = self.dataset.indices[idx]
            image, label = self.dataset.dataset[original_idx]
        else:
            image, label = self.dataset[idx]
            original_idx = idx
        
        # 确保image是tensor格式
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        # 生成双视图
        weak_view = self.weak_aug(image)
        strong_view = self.strong_aug(image)
        
        if self.return_index:
            return weak_view, strong_view, label, original_idx
        else:
            return weak_view, strong_view, label


class IndexedDataset(Dataset):
    """带索引的数据集包装器，便于索引映射"""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        # 构建全局索引映射
        if hasattr(dataset, 'indices'):
            self.global_indices = dataset.indices
        else:
            self.global_indices = list(range(len(dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        global_idx = self.global_indices[idx]
        return (*data, global_idx)


# ============================================================================
# 3. 模型架构（支持OOD检测）
# ============================================================================

class WiseOpenModel(nn.Module):
    """WiseOpen模型，包含分类头和OOD检测头"""
    
    def __init__(self, backbone: nn.Module, num_classes: int, 
                 feature_dim: int = 512, ood_head: bool = True):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ood_head_enabled = ood_head
        
        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # OOD检测头（二分类：ID vs OOD）
        if ood_head:
            self.ood_head = nn.Linear(feature_dim, 1)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        
        # 分类输出
        logits = self.classifier(features)
        
        results = {'logits': logits}
        
        # OOD检测输出
        if self.ood_head_enabled:
            ood_score = self.ood_head(features)
            results['ood_score'] = ood_score
        
        if return_features:
            results['features'] = features
            
        return results


def create_resnet18_model(num_classes: int, pretrained: bool = True) -> WiseOpenModel:
    """创建ResNet18模型"""
    backbone = torchvision.models.resnet18(pretrained=pretrained)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()  # 移除最后的分类层
    
    return WiseOpenModel(backbone, num_classes, feature_dim)


# ============================================================================
# 4. 无监督损失函数
# ============================================================================

class FixMatchLoss(nn.Module):
    """FixMatch风格的无监督损失"""
    
    def __init__(self, threshold: float = 0.95, temperature: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, weak_logits: torch.Tensor, strong_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weak_logits: 弱增强样本的logits [N, C]
            strong_logits: 强增强样本的logits [N, C]
        """
        # 生成伪标签
        with torch.no_grad():
            weak_probs = F.softmax(weak_logits / self.temperature, dim=1)
            max_probs, pseudo_labels = torch.max(weak_probs, dim=1)
            
            # 置信度掩码
            mask = max_probs >= self.threshold
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=weak_logits.device, requires_grad=True)
        
        # 计算强增强样本的交叉熵损失
        strong_loss = self.ce_loss(strong_logits, pseudo_labels)
        
        # 只对高置信度样本计算损失
        masked_loss = strong_loss * mask.float()
        
        return masked_loss.mean()


class ConsistencyLoss(nn.Module):
    """一致性损失"""
    
    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, weak_logits: torch.Tensor, strong_logits: torch.Tensor) -> torch.Tensor:
        weak_probs = F.softmax(weak_logits, dim=1)
        strong_probs = F.softmax(strong_logits, dim=1)
        
        # 置信度掩码
        max_probs, _ = torch.max(weak_probs, dim=1)
        mask = max_probs >= self.threshold
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=weak_logits.device, requires_grad=True)
        
        # MSE损失
        consistency_loss = self.mse_loss(strong_probs, weak_probs).sum(dim=1)
        masked_loss = consistency_loss * mask.float()
        
        return masked_loss.mean()


# ============================================================================
# 5. OOD预过滤器
# ============================================================================

class OODPrefilter:
    """OOD样本预过滤器"""
    
    def __init__(self, model: WiseOpenModel, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
    
    def filter_ood_samples(self, dataloader: DataLoader, device: torch.device) -> List[int]:
        """过滤明显的OOD样本，返回ID样本的索引"""
        self.model.eval()
        id_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 4:  # (weak, strong, label, index)
                    weak_imgs, _, _, indices = batch
                else:  # (weak, strong, label)
                    weak_imgs = batch[0]
                    indices = list(range(batch_idx * dataloader.batch_size, 
                                       batch_idx * dataloader.batch_size + len(weak_imgs)))
                
                weak_imgs = weak_imgs.to(device)
                outputs = self.model(weak_imgs)
                
                if 'ood_score' in outputs:
                    # 使用OOD头进行过滤
                    ood_probs = torch.sigmoid(outputs['ood_score']).squeeze()
                    id_mask = ood_probs < self.threshold
                else:
                    # 使用最大预测概率进行过滤
                    probs = F.softmax(outputs['logits'], dim=1)
                    max_probs, _ = torch.max(probs, dim=1)
                    id_mask = max_probs > self.threshold
                
                # 收集ID样本索引
                batch_id_indices = [indices[i] for i in range(len(indices)) if id_mask[i]]
                id_indices.extend(batch_id_indices)
        
        self.model.train()
        return id_indices


# ============================================================================
# 6. 改进的选择机制
# ============================================================================

def efficient_gvsm_selection(model: nn.Module, 
                           labeled_batch: Tuple[torch.Tensor, torch.Tensor],
                           unlabeled_batch: Tuple[torch.Tensor, torch.Tensor],
                           supervised_loss_fn: nn.Module,
                           unsupervised_loss_fn: nn.Module,
                           selection_ratio: float,
                           device: torch.device,
                           last_layers_only: bool = True) -> torch.Tensor:
    """
    高效的梯度方差选择机制（只计算最后几层的梯度）
    """
    model.eval()
    
    labeled_images, labeled_targets = labeled_batch
    unlabeled_weak, unlabeled_strong, _, _ = unlabeled_batch  
    batch_size = unlabeled_weak.size(0)
    k = int(selection_ratio * batch_size)
    
    if k == 0:
        return torch.tensor([], dtype=torch.long)
    
    # 选择要计算梯度的参数（只选择最后几层）
    if last_layers_only:
        target_params = []
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name or 'head' in name:
                target_params.append(param)
        if not target_params:
            target_params = list(model.parameters())[-4:]  # 最后4层
    else:
        target_params = list(model.parameters())
    
    # 步骤1: 计算平均梯度
    model.zero_grad()
    labeled_logits = model(labeled_images.to(device))['logits']
    supervised_loss = supervised_loss_fn(labeled_logits, labeled_targets.to(device))
    supervised_loss.backward()
    
    # 收集平均梯度
    g_bar = []
    for param in target_params:
        if param.grad is not None:
            g_bar.append(param.grad.view(-1).clone())
    g_bar = torch.cat(g_bar) if g_bar else torch.tensor([])
    
    # 步骤2: 批量计算梯度方差（使用functorch或手动实现）
    gradient_variances = []
    
    # 简化版本：使用预测不确定性作为代理
    with torch.no_grad():
        weak_outputs = model(unlabeled_weak.to(device))
        weak_probs = F.softmax(weak_outputs['logits'], dim=1)
        
        # 使用预测熵作为不确定性度量
        entropy = -torch.sum(weak_probs * torch.log(weak_probs + 1e-8), dim=1)
        gradient_variances = entropy
    
    # 选择不确定性最低的样本（更可能是友好样本）
    _, selected_indices = torch.topk(gradient_variances, k, largest=False)
    
    model.train()
    return selected_indices


def improved_lsm_selection(model: nn.Module,
                         unlabeled_batch: Tuple[torch.Tensor, torch.Tensor],
                         unsupervised_loss_fn: nn.Module,
                         selection_ratio: float,
                         device: torch.device) -> torch.Tensor:
    """改进的基于损失的选择机制"""
    model.eval()
    
    unlabeled_weak, unlabeled_strong, _, _ = unlabeled_batch

    batch_size = unlabeled_weak.size(0)
    k = int(selection_ratio * batch_size)
    
    if k == 0:
        return torch.tensor([], dtype=torch.long)
    
    with torch.no_grad():
        weak_outputs = model(unlabeled_weak.to(device))
        strong_outputs = model(unlabeled_strong.to(device))
        
        weak_logits = weak_outputs['logits']
        strong_logits = strong_outputs['logits']
        
        # 计算一致性损失或FixMatch损失
        if hasattr(unsupervised_loss_fn, 'ce_loss'):
            # FixMatch损失
            weak_probs = F.softmax(weak_logits, dim=1)
            max_probs, pseudo_labels = torch.max(weak_probs, dim=1)
            sample_losses = F.cross_entropy(strong_logits, pseudo_labels, reduction='none')
            
            # 结合置信度
            confidence_scores = max_probs
            combined_scores = sample_losses / (confidence_scores + 1e-8)
        else:
            # 一致性损失
            weak_probs = F.softmax(weak_logits, dim=1)
            strong_probs = F.softmax(strong_logits, dim=1)
            sample_losses = F.mse_loss(strong_probs, weak_probs, reduction='none').sum(dim=1)
            combined_scores = sample_losses
    
    # 选择损失最小的样本
    _, selected_indices = torch.topk(combined_scores, k, largest=False)
    
    model.train()
    return selected_indices


# ============================================================================
# 7. 友好样本管理器
# ============================================================================

class FriendlySampleManager:
    """友好样本管理器"""
    
    def __init__(self, selection_memory: int = 3, min_frequency: int = 2):
        self.selection_memory = selection_memory
        self.min_frequency = min_frequency
        self.selection_history = []
        self.sample_frequencies = Counter()
    
    def update_selections(self, selected_indices: List[int]):
        """更新选择历史"""
        self.selection_history.append(set(selected_indices))
        
        # 保持历史记录长度
        if len(self.selection_history) > self.selection_memory:
            old_selection = self.selection_history.pop(0)
            # 减少旧选择的频次
            for idx in old_selection:
                self.sample_frequencies[idx] -= 1
                if self.sample_frequencies[idx] <= 0:
                    del self.sample_frequencies[idx]
        
        # 增加新选择的频次
        for idx in selected_indices:
            self.sample_frequencies[idx] += 1
    
    def get_friendly_indices(self) -> List[int]:
        """获取友好样本索引"""
        return [idx for idx, freq in self.sample_frequencies.items() 
                if freq >= self.min_frequency]
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        friendly_indices = self.get_friendly_indices()
        return {
            'total_candidates': len(self.sample_frequencies),
            'friendly_samples': len(friendly_indices),
            'avg_frequency': np.mean(list(self.sample_frequencies.values())) if self.sample_frequencies else 0,
            'max_frequency': max(self.sample_frequencies.values()) if self.sample_frequencies else 0
        }


# ============================================================================
# 8. 主训练循环
# ============================================================================

def wiseopen_training_loop(model: WiseOpenModel,
                          labeled_loader: DataLoader,
                          unlabeled_loader: DataLoader,
                          val_loader: DataLoader,
                          optimizer: optim.Optimizer,
                          scheduler: optim.lr_scheduler._LRScheduler,
                          sup_loss_fn: nn.Module,
                          unsup_loss_fn: nn.Module,
                          selection_fn: callable,
                          num_epochs: int,
                          e_s: int,
                          selection_ratio: float,
                          device: torch.device,
                          lambda_u: float = 1.0,
                          lambda_ood: float = 0.1,
                          use_ood_filter: bool = True,
                          logger: logging.Logger = None):
    """完整的WiseOpen训练循环"""
    
    model.to(device)
    
    # 初始化组件
    friendly_manager = FriendlySampleManager()
    ood_filter = OODPrefilter(model) if use_ood_filter else None
    
    # 训练历史
    train_history = {
        'train_loss': [], 'sup_loss': [], 'unsup_loss': [], 'ood_loss': [],
        'val_acc': [], 'friendly_count': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        if logger:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # 数据选择阶段
        if epoch % e_s == 0:
            if logger:
                logger.info(f"执行数据选择 (epoch {epoch + 1})")
            
            # OOD预过滤
            if ood_filter and epoch > 0:
                id_indices = ood_filter.filter_ood_samples(unlabeled_loader, device)
                if logger:
                    logger.info(f"OOD预过滤后剩余 {len(id_indices)} 个样本")
            
            # 选择友好样本
            all_selected_indices = []
            selection_batches = min(10, len(unlabeled_loader))
            
            for batch_idx, unlabeled_batch in enumerate(unlabeled_loader):
                if batch_idx >= selection_batches:
                    break
                
                # 获取有标签批次
                labeled_batch = next(iter(labeled_loader))
                
                # 执行选择
                if selection_fn == efficient_gvsm_selection:
                    selected_indices = selection_fn(
                        model, labeled_batch, unlabeled_batch,
                        sup_loss_fn, unsup_loss_fn, selection_ratio, device
                    )
                else:
                    selected_indices = selection_fn(
                        model, unlabeled_batch, unsup_loss_fn, selection_ratio, device
                    )
                
                # 转换为全局索引
                if len(unlabeled_batch) == 4:  # 包含索引
                    global_indices = [unlabeled_batch[3][i].item() for i in selected_indices]
                else:
                    batch_start = batch_idx * unlabeled_loader.batch_size
                    global_indices = [batch_start + idx.item() for idx in selected_indices]
                
                all_selected_indices.extend(global_indices)
            
            # 更新友好样本管理器
            friendly_manager.update_selections(all_selected_indices)
            friendly_indices = friendly_manager.get_friendly_indices()
            
            stats = friendly_manager.get_selection_stats()
            if logger:
                logger.info(f"选择统计: {stats}")
            
            train_history['friendly_count'].append(len(friendly_indices))
        
        # 训练阶段
        model.train()
        epoch_stats = {'total_loss': 0, 'sup_loss': 0, 'unsup_loss': 0, 'ood_loss': 0, 'batches': 0}
        
        # 创建友好样本数据加载器
        if len(friendly_indices) > 0:
            # 1. 从加载器中追溯到最原始的数据集
            #    unlabeled_loader.dataset -> DualViewDataset
            #    unlabeled_loader.dataset.dataset -> unlabeled_dataset (Subset)
            #    unlabeled_loader.dataset.dataset.dataset -> 原始的CIFAR10训练集
            original_train_dataset = unlabeled_loader.dataset.dataset.dataset

            # 2. 使用全局索引，从最原始的数据集中创建正确的子集
            friendly_subset = Subset(original_train_dataset, friendly_indices)

            # 3. 为这个新的、正确的子集应用双视图增强，并创建加载器
            friendly_loader = DataLoader(
                DualViewDataset(friendly_subset, return_index=True), # 显式添加参数
                batch_size=labeled_loader.batch_size,
                shuffle=True,
                num_workers=2
            )
            friendly_iter = iter(friendly_loader)
        else:
            friendly_iter = None
        
        for batch_idx, (labeled_images, labeled_targets) in enumerate(labeled_loader):
            labeled_images = labeled_images.to(device)
            labeled_targets = labeled_targets.to(device)
            
            # 监督损失
            labeled_outputs = model(labeled_images)
            sup_loss = sup_loss_fn(labeled_outputs['logits'], labeled_targets)
            total_loss = sup_loss
            
            # 无监督损失
            unsup_loss = torch.tensor(0.0, device=device)
            if friendly_iter is not None:
                try:
                    unlabeled_weak, unlabeled_strong, _, _ = next(friendly_iter)
                    unlabeled_weak = unlabeled_weak.to(device)
                    unlabeled_strong = unlabeled_strong.to(device)
                    
                    weak_outputs = model(unlabeled_weak)
                    strong_outputs = model(unlabeled_strong)
                    
                    unsup_loss = unsup_loss_fn(weak_outputs['logits'], strong_outputs['logits'])
                    total_loss += lambda_u * unsup_loss
                    
                except StopIteration:
                    friendly_iter = iter(friendly_loader) if len(friendly_indices) > 0 else None
            
            # OOD损失
            ood_loss = torch.tensor(0.0, device=device)
            if model.ood_head_enabled and epoch > 0:
                # 简单的OOD损失：ID样本应该被预测为非OOD
                ood_logits = labeled_outputs['ood_score']
                ood_targets = torch.zeros_like(ood_logits.squeeze())  # ID样本标签为0
                ood_loss = F.binary_cross_entropy_with_logits(ood_logits.squeeze(), ood_targets)
                total_loss += lambda_ood * ood_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            epoch_stats['total_loss'] += total_loss.item()
            epoch_stats['sup_loss'] += sup_loss.item()
            epoch_stats['unsup_loss'] += unsup_loss.item()
            epoch_stats['ood_loss'] += ood_loss.item()
            epoch_stats['batches'] += 1
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练损失
        for key in ['total_loss', 'sup_loss', 'unsup_loss', 'ood_loss']:
            avg_loss = epoch_stats[key] / epoch_stats['batches']
            train_history[key.replace('total_', 'train_')].append(avg_loss)
        
        # 验证
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_acc = evaluate_classification(model, val_loader, device)
            train_history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
            
            if logger:
                logger.info(f"验证准确率: {val_acc:.4f}, 最佳: {best_val_acc:.4f}")
        
        # 打印训练信息
        if logger and (epoch + 1) % 1 == 0:
            logger.info(f"损失 - 总计: {train_history['train_loss'][-1]:.4f}, "
                       f"监督: {train_history['sup_loss'][-1]:.4f}, "
                       f"无监督: {train_history['unsup_loss'][-1]:.4f}")
    
    return train_history


# ============================================================================
# 9. 评估函数
# ============================================================================

def evaluate_classification(model: WiseOpenModel, dataloader: DataLoader, 
                          device: torch.device) -> float:
    """评估ID分类准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.train()
    return correct / total


def evaluate_ood_detection(model: WiseOpenModel, id_loader: DataLoader, 
                          ood_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """评估OOD检测性能"""
    model.eval()
    
    id_scores = []
    ood_scores = []
    
    with torch.no_grad():
        # 收集ID样本的OOD分数
        for batch in id_loader:
            if len(batch) == 2:
                images = batch[0]
            else:
                images = batch[0]
            
            images = images.to(device)
            outputs = model(images)
            
            if 'ood_score' in outputs:
                scores = torch.sigmoid(outputs['ood_score']).cpu().numpy().flatten()
            else:
                # 使用最大预测概率作为ID分数
                probs = F.softmax(outputs['logits'], dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                scores = 1 - max_probs.cpu().numpy()  # 转换为OOD分数
            
            id_scores.extend(scores)
        
        # 收集OOD样本的OOD分数
        for batch in ood_loader:
            if len(batch) == 2:
                images = batch[0]
            else:
                images = batch[0]
            
            images = images.to(device)
            outputs = model(images)
            
            if 'ood_score' in outputs:
                scores = torch.sigmoid(outputs['ood_score']).cpu().numpy().flatten()
            else:
                probs = F.softmax(outputs['logits'], dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                scores = 1 - max_probs.cpu().numpy()
            
            ood_scores.extend(scores)
    
    # 计算AUROC
    y_true = [0] * len(id_scores) + [1] * len(ood_scores)
    y_score = id_scores + ood_scores
    
    auroc = roc_auc_score(y_true, y_score)
    
    model.train()
    return {
        'auroc': auroc,
        'id_scores_mean': np.mean(id_scores),
        'ood_scores_mean': np.mean(ood_scores)
    }


# ============================================================================
# 10. 数据集划分器（原有代码的改进版）
# ============================================================================

class OSSLDatasetSplitter:
    """开放集半监督学习数据集划分器"""
    
    def __init__(self, dataset: Dataset, seen_classes: List[int], 
                 n_labeled_per_class: int, n_val_per_class: int,
                 random_seed: int = 42):
        self.dataset = dataset
        self.seen_classes = set(seen_classes)
        self.n_labeled_per_class = n_labeled_per_class
        self.n_val_per_class = n_val_per_class
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 构建类别索引映射
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            self.class_to_indices[label].append(idx)
        
        # 获取所有类别
        self.all_classes = set(self.class_to_indices.keys())
        self.unseen_classes = self.all_classes - self.seen_classes
    
    def split_dataset(self) -> Tuple[Dataset, Dataset, Dataset, Dataset, List[int], List[int]]:
        """划分数据集"""
        labeled_indices = []
        val_indices = []
        unlabeled_indices = []
        test_indices = []
        
        # 处理已知类别
        for class_id in self.seen_classes:
            class_indices = self.class_to_indices[class_id].copy()
            random.shuffle(class_indices)
            
            required_samples = self.n_labeled_per_class + self.n_val_per_class
            if len(class_indices) < required_samples:
                print(f"警告: 类别 {class_id} 样本不足")
                continue
            
            # 分配样本
            labeled_indices.extend(class_indices[:self.n_labeled_per_class])
            val_start = self.n_labeled_per_class
            val_end = val_start + self.n_val_per_class
            val_indices.extend(class_indices[val_start:val_end])
            
            # 剩余样本分为unlabeled和test
            remaining = class_indices[val_end:]
            split_point = len(remaining) // 2
            unlabeled_indices.extend(remaining[:split_point])
            test_indices.extend(remaining[split_point:])
        
        # 处理未知类别
        for class_id in self.unseen_classes:
            class_indices = self.class_to_indices[class_id]
            split_point = len(class_indices) // 2
            unlabeled_indices.extend(class_indices[:split_point])
            test_indices.extend(class_indices[split_point:])
        
        # 创建数据集
        labeled_dataset = Subset(self.dataset, labeled_indices)
        unlabeled_dataset = Subset(self.dataset, unlabeled_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)
        
        return (labeled_dataset, unlabeled_dataset, val_dataset, test_dataset,
                list(self.seen_classes), list(self.unseen_classes))


# ============================================================================
# 11. 完整使用示例
# ============================================================================

def setup_logging() -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wiseopen_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_data_loaders(dataset_name: str = 'CIFAR10', batch_size: int = 128, 
                       num_workers: int = 2) -> Tuple[Dataset, Dataset]:
    """创建数据集"""
    # CIFAR-10标准数据增强
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
    elif dataset_name == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def plot_training_history(history: Dict[str, List], save_path: str = 'training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Total Loss')
    axes[0, 0].plot(history['sup_loss'], label='Supervised Loss')
    axes[0, 0].plot(history['unsup_loss'], label='Unsupervised Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证准确率
    if history['val_acc']:
        val_epochs = [i * 5 for i in range(len(history['val_acc']))]
        axes[0, 1].plot(val_epochs, history['val_acc'], 'o-', label='Validation Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 友好样本数量
    if history['friendly_count']:
        selection_epochs = [i * 5 for i in range(len(history['friendly_count']))]
        axes[1, 0].plot(selection_epochs, history['friendly_count'], 's-', label='Friendly Samples')
        axes[1, 0].set_title('Friendly Sample Count')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # OOD损失
    if history['ood_loss']:
        axes[1, 1].plot(history['ood_loss'], label='OOD Loss')
        axes[1, 1].set_title('OOD Detection Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数 - 完整的WiseOpen训练示例"""
    
    # ============================================================================
    # 1. 配置和初始化
    # ============================================================================
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始WiseOpen训练")
    
    # ============================================================================
    # 2. 数据准备
    # ============================================================================
    
    # 创建数据集
    train_dataset, test_dataset = create_data_loaders('CIFAR10')
    
    # 定义已知类别（CIFAR-10前6类作为已知类别）
    seen_classes = [0, 1, 2, 3, 4, 5]
    num_seen_classes = len(seen_classes)
    
    # 数据集划分
    splitter = OSSLDatasetSplitter(
        dataset=train_dataset,
        seen_classes=seen_classes,
        n_labeled_per_class=100,  # 每类100个标签样本
        n_val_per_class=50        # 每类50个验证样本
    )
    
    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset_split, \
    seen_classes_list, unseen_classes_list = splitter.split_dataset()
    
    logger.info(f"已知类别: {seen_classes_list}")
    logger.info(f"未知类别: {unseen_classes_list}")
    logger.info(f"标签样本: {len(labeled_dataset)}")
    logger.info(f"无标签样本: {len(unlabeled_dataset)}")
    logger.info(f"验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    
    unlabeled_loader = DataLoader(
        DualViewDataset(unlabeled_dataset, return_index=True), 
        batch_size=128, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    
    # 创建测试集（用于最终评估）
    # ID测试集
    id_test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                      if label in seen_classes]
    id_test_dataset = Subset(test_dataset, id_test_indices)
    id_test_loader = DataLoader(id_test_dataset, batch_size=128, shuffle=False)
    
    # OOD测试集
    ood_test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                       if label in unseen_classes_list]
    ood_test_dataset = Subset(test_dataset, ood_test_indices)
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=128, shuffle=False)
    
    logger.info(f"ID测试样本: {len(id_test_dataset)}")
    logger.info(f"OOD测试样本: {len(ood_test_dataset)}")
    
    # ============================================================================
    # 3. 模型创建
    # ============================================================================
    
    model = create_resnet18_model(num_classes=num_seen_classes, pretrained=True)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # ============================================================================
    # 4. 损失函数和优化器
    # ============================================================================
    
    # 损失函数
    sup_loss_fn = nn.CrossEntropyLoss()
    unsup_loss_fn = FixMatchLoss(threshold=0.95)
    
    # 优化器和调度器
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.03, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6
    )
    
    # ============================================================================
    # 5. 超参数设置
    # ============================================================================
    
    training_config = {
        'num_epochs': 200,
        'e_s': 5,                    # 每5个epoch进行一次选择
        'selection_ratio': 0.3,      # 选择30%的样本
        'lambda_u': 1.0,            # 无监督损失权重
        'lambda_ood': 0.1,          # OOD损失权重
        'use_ood_filter': True,     # 使用OOD预过滤
    }
    
    logger.info(f"训练配置: {training_config}")
    
    # ============================================================================
    # 6. 训练
    # ============================================================================
    
    logger.info("开始训练...")
    
    train_history = wiseopen_training_loop(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        sup_loss_fn=sup_loss_fn,
        unsup_loss_fn=unsup_loss_fn,
        selection_fn=improved_lsm_selection,  # 可以选择 efficient_gvsm_selection
        device=device,
        logger=logger,
        **training_config
    )
    
    logger.info("训练完成!")
    
    # ============================================================================
    # 7. 最终评估
    # ============================================================================
    
    # 加载最佳模型
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        logger.info("加载最佳模型进行评估")
    except:
        logger.warning("未找到最佳模型文件，使用当前模型")
    
    # ID分类性能评估
    id_acc = evaluate_classification(model, id_test_loader, device)
    logger.info(f"ID分类准确率: {id_acc:.4f}")
    
    # OOD检测性能评估
    ood_results = evaluate_ood_detection(model, id_test_loader, ood_test_loader, device)
    logger.info(f"OOD检测AUROC: {ood_results['auroc']:.4f}")
    logger.info(f"ID样本平均OOD分数: {ood_results['id_scores_mean']:.4f}")
    logger.info(f"OOD样本平均OOD分数: {ood_results['ood_scores_mean']:.4f}")
    
    # ============================================================================
    # 8. 结果可视化
    # ============================================================================
    
    plot_training_history(train_history)
    
    # 保存最终结果
    final_results = {
        'id_accuracy': id_acc,
        'ood_auroc': ood_results['auroc'],
        'training_config': training_config,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    logger.info("=== 最终结果 ===")
    for key, value in final_results.items():
        logger.info(f"{key}: {value}")
    
    return model, train_history, final_results


# ============================================================================
# 12. 超参数配置建议
# ============================================================================

def get_recommended_hyperparameters(dataset_name: str) -> Dict[str, Any]:
    """获取推荐的超参数配置"""
    
    configs = {
        'CIFAR10': {
            'learning_rate': 0.03,
            'batch_size': 128,
            'num_epochs': 200,
            'selection_interval': 5,
            'selection_ratio': 0.3,
            'lambda_u': 1.0,
            'lambda_ood': 0.1,
            'fixmatch_threshold': 0.95,
            'weight_decay': 5e-4,
            'n_labeled_per_class': 100,
            'n_val_per_class': 50
        },
        'CIFAR100': {
            'learning_rate': 0.02,
            'batch_size': 64,
            'num_epochs': 300,
            'selection_interval': 10,
            'selection_ratio': 0.2,
            'lambda_u': 2.0,
            'lambda_ood': 0.05,
            'fixmatch_threshold': 0.9,
            'weight_decay': 1e-3,
            'n_labeled_per_class': 50,
            'n_val_per_class': 25
        }
    }
    
    return configs.get(dataset_name, configs['CIFAR10'])


if __name__ == "__main__":
    # 运行完整的训练和评估流程
    model, history, results = main()