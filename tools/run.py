
import tkinter as tk

from tkinter import filedialog
from tkinter import messagebox
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import src.fridge_eval.config as config


root = tk.Tk()
root.withdraw()

input_file = filedialog.askopenfilename(
    title="请选择Excel文件",
    filetypes=[("Excel Files", "*.xlsx")]
)

if not input_file:
    raise SystemExit
from src.fridge_eval.pipeline_with_juice_complete import run_pipeline
base = os.path.splitext(input_file)[0]

config.OUTPUT_FILE = (
    base + "_result.xlsx"
)

try:

    run_pipeline(input_file)


    result_file = os.path.abspath(
        config.OUTPUT_FILE
    )
    messagebox.showinfo(
        "完成",
        f"结果已生成：\n\n结果文件位置：\n{result_file}"
    )


except Exception as e:

    messagebox.showerror(
        "错误",
        str(e)
    )