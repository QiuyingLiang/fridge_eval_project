import pandas as pd
import os
from pipeline.detector import Detector
from pipeline.executor import run_parallel
from pipeline.worker import process_one
from evaluation import evaluate, print_evaluation_report
from config import INPUT_EXCEL, OUTPUT_EXCEL

def main():
    # 读取数据
    df = pd.read_excel(INPUT_EXCEL)
    
    # 清理列名
    df.columns = df.columns.str.strip()
    
    print(f"原始数据: {len(df)} 行")
    print(f"列名: {df.columns.tolist()}")
    
    # 新增列
    df["m_layer"] = 0
    df["m_problem"] = ""
    df["是否准确"] = ""
    
    # 初始化检测器
    detector = Detector()
    
    # 并行处理
    results = run_parallel(df, detector, process_one)
    
    # 更新结果
    for result in results:
        if len(result) == 5:  # 新格式
            idx, m_layer, m_problem, debug_info, detail = result
        else:  # 兼容旧格式
            idx, fridge, bottle, status, m_layer, m_problem = result
        
        df.loc[idx, "m_layer"] = m_layer
        df.loc[idx, "m_problem"] = m_problem
    
    # 评估
    df["是否准确"] = df.apply(evaluate, axis=1)
    
    # 保存结果
    os.makedirs("output", exist_ok=True)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n结果已保存到: {OUTPUT_EXCEL}")
    
    # 打印评估报告
    print_evaluation_report(df)

if __name__ == "__main__":
    main()