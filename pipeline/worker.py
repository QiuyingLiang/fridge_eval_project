from pipeline.loader import load_image
from config import TARGET_BRAND, MIN_ROWS_PER_LAYER
import numpy as np

def process_one(idx, row, detector):
    """
    返回: (idx, m_layer, m_problem, debug_info, detail)
    - m_layer: 符合标准的层数（排数 >= MIN_ROWS_PER_LAYER 且全部是脉动）
    - m_problem: 问题类型（正常/纯度不足-空心萝卜/纯度不足-花心萝卜）
    """
    url = row['image_url']
    img = load_image(url)
    
    if img is None:
        return idx, 0, '图片翻拍', 'load_failed', '图片加载失败'
    
    # 检测所有瓶子
    bottles = detector.detect_bottles_with_position(img)
    
    # 情况1: 没有瓶子
    if len(bottles) == 0:
        return idx, 0, '冰柜是空柜', 'empty', '无瓶子'
    
    # 检查是否有非脉动品牌（花心萝卜）
    has_other_brand = any(b['brand'] != '脉动' for b in bottles)
    
    if has_other_brand:
        return idx, 0, '纯度不足-花心萝卜', 'mixed_brand', '存在非脉动品牌'
    
    # 分析每层的排数
    layers = analyze_layers_by_y_position(bottles, img.shape[0])
    
    # 统计符合标准的层数（排数 >= 2 且全部是脉动）
    valid_layers = 0
    has_insufficient_rows = False
    
    for layer_id, layer_info in layers.items():
        rows = layer_info['rows']
        if rows >= MIN_ROWS_PER_LAYER:
            valid_layers += 1
        else:
            has_insufficient_rows = True
    
    # 情况2: 有层排数不足（空心萝卜）
    if has_insufficient_rows:
        return idx, valid_layers, '纯度不足-空心萝卜', 'insufficient_rows', f'{valid_layers}层符合标准，{len(layers)-valid_layers}层排数不足'
    
    # 情况3: 全部正常
    return idx, valid_layers, '正常', 'ok', f'{valid_layers}层全部符合标准'

def analyze_layers_by_y_position(bottles, image_height):
    """
    根据瓶子的Y坐标进行分层，并计算每层的排数
    假设冰箱有4层
    """
    if len(bottles) == 0:
        return {}
    
    num_layers = 4
    layer_height = image_height / num_layers
    
    # 初始化每层
    layers = {}
    for i in range(num_layers):
        layers[i+1] = {
            'bottles': [],
            'y_min': i * layer_height,
            'y_max': (i+1) * layer_height,
            'rows': 0
        }
    
    # 分配瓶子到各层
    for bottle in bottles:
        y = bottle['y']
        layer_id = min(int(y // layer_height), num_layers - 1) + 1
        layers[layer_id]['bottles'].append(bottle)
    
    # 计算每层的排数
    for layer_id, layer_info in layers.items():
        bottles_in_layer = layer_info['bottles']
        if len(bottles_in_layer) == 0:
            layer_info['rows'] = 0
            continue
        
        # 根据Y坐标聚类计算排数
        y_coords = [b['y'] for b in bottles_in_layer]
        y_coords.sort()
        
        rows = 1
        y_threshold = 30  # 像素阈值
        
        for i in range(1, len(y_coords)):
            if y_coords[i] - y_coords[i-1] > y_threshold:
                rows += 1
        
        layer_info['rows'] = rows
    
    return layers