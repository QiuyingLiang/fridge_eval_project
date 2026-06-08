import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.fridge_eval.pipeline import run_pipeline

if __name__=='__main__':
    run_pipeline('data/input.xlsx')
