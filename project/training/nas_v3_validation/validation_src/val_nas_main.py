import argparse
import os

from val_nas_executor import NASExecutor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS V3 Validation')
    parser.add_argument('--algorithm',       type=str, default='random',  help='algorithm to use for searching')
    parser.add_argument('--dataset',         type=str, default='cifar10', help='dataset to use for searching')
    parser.add_argument('--max_train_hours', type=int, default=2e4,       help='max training hours for searching')
    parser.add_argument('--reporting_epoch', type=int, default=12,        help='reporting epoch for searching')
    parser.add_argument('--ss_indicator',    type=str, default='sss',     help='search space indicator to identify the search space')
    parser.add_argument('--output_dir',      type=str, default='data',    help='output directory for saving results')
    args = parser.parse_args()

    print(args)

    executor = NASExecutor(args.algorithm, args.dataset, args.max_train_hours, args.reporting_epoch, args.ss_indicator)
    results_df = executor.test_nas_algo()

    #output_filename = os.path.join(args.output_dir, f'{args.algorithm}_{args.dataset}_{args.max_train_hours}.csv')
    #val_nas_main.save_report(results_df, output_filename)