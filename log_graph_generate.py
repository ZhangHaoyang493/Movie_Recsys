import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Graph Log Generation")
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to input data")
    parser.add_argument("--column", "-c", type=str, required=True, help="The name of the column to visualize")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    column = args.column

    basedir = os.path.dirname(input_path)

    with open(input_path, 'r') as f:
        lines = f.readlines()
    values, steps = [], []
    columns = lines[0].strip().split(',')
    col_idx = columns.index(column)
    step_idx = columns.index('step')
    for line in lines[1:]:
        value = line.strip().split(',')[col_idx]
        step = line.strip().split(',')[step_idx]
        if value != '':
            values.append(float(value))
            steps.append(int(step))

    # 生成日志图，横轴为steps，纵轴为values
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=column)
    plt.xlabel('Steps')
    plt.ylabel(column)
    plt.title(f'Log Graph of {column}')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(basedir, f'{column}.png')
    plt.savefig(output_path)
    print(f'Log graph saved to {output_path}')
    