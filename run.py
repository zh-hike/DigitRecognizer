import argparse
from Trainer import Trainer



parse = argparse.ArgumentParser('Digit Recognizer')

parse.add_argument('--batch_size', type=int, help="批次大小", default=100000)
parse.add_argument('--epochs', type=int, help="迭代次数", default=100)
parse.add_argument('--data_path', type=str, help="数据集路径", default='../../MyData/competition/mnist')

args = parse.parse_args()

if __name__ == "__main__":
    trainer = Trainer(args)

    trainer.train()
