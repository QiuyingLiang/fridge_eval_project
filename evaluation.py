import pandas as pd

def compare_layer(gt_layer, m_layer):
    """比较层数，允许±1误差"""
    try:
        # 处理空值
        if pd.isna(gt_layer) or gt_layer == '':
            return None  # 无人工标注，不评估
        gt_val = int(float(gt_layer))
        m_val = int(m_layer) if not pd.isna(m_layer) else 0
        return abs(gt_val - m_val) <= 1
    except:
        return False

def compare_problem(gt_problem, m_problem):
    """比较问题类型"""
    # 处理空值
    if pd.isna(gt_problem) or gt_problem == '':
        return None  # 无人工标注，不评估
    
    gt_str = str(gt_problem).strip()
    m_str = str(m_problem).strip() if not pd.isna(m_problem) else ''
    
    # 只评估两种纯度问题
    valid_problems = ['纯度不足-空心萝卜', '纯度不足-花心萝卜', '正常']
    
    # 如果机器输出不在评估范围内，标记为错误
    if m_str not in valid_problems:
        return False
    
    return gt_str == m_str

def evaluate(row):
    """
    评估单行结果
    只评估有人工标注的行
    """
    gt_layer = row.get('pt_layer')
    gt_problem = row.get('pt_problem')
    
    # 如果人工标注都没有数据，跳过评估
    if pd.isna(gt_layer) and (pd.isna(gt_problem) or gt_problem == ''):
        return 'skip'
    
    layer_ok = compare_layer(gt_layer, row.get('m_layer'))
    problem_ok = compare_problem(gt_problem, row.get('m_problem'))
    
    # 如果层数没有人工标注，只评估问题
    if layer_ok is None:
        if problem_ok is None:
            return 'skip'
        return 'correct' if problem_ok else 'wrong'
    
    # 如果问题没有人工标注，只评估层数
    if problem_ok is None:
        return 'correct' if layer_ok else 'wrong'
    
    # 两者都有标注
    if layer_ok and problem_ok:
        return 'correct'
    else:
        return 'wrong'

def print_evaluation_report(df):
    """打印评估报告"""
    # 过滤出有评估的行
    eval_df = df[df['是否准确'] != 'skip'].copy()
    
    if len(eval_df) == 0:
        print("没有可评估的数据（缺少人工标注）")
        return
    
    total = len(eval_df)
    correct = (eval_df['是否准确'] == 'correct').sum()
    
    print(f"\n{'='*50}")
    print(f"评估报告")
    print(f"{'='*50}")
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {correct/total:.2%}")
    
    # 按问题类型统计
    print(f"\n问题类型统计:")
    print(f"{'问题类型':<20} {'数量':<10} {'正确率':<10}")
    print(f"{'-'*40}")
    
    for problem in ['纯度不足-空心萝卜', '纯度不足-花心萝卜', '正常']:
        subset = eval_df[eval_df['m_problem'] == problem]
        if len(subset) > 0:
            acc = (subset['是否准确'] == 'correct').mean()
            print(f"{problem:<20} {len(subset):<10} {acc:.2%}")
    
    # 层数准确率统计
    print(f"\n层数准确率:")
    layer_correct = 0
    layer_total = 0
    for _, row in eval_df.iterrows():
        if pd.notna(row.get('pt_layer')) and row.get('pt_layer') != '':
            layer_total += 1
            if compare_layer(row['pt_layer'], row.get('m_layer')):
                layer_correct += 1
    
    if layer_total > 0:
        print(f"层数准确率: {layer_correct/layer_total:.2%} ({layer_correct}/{layer_total})")