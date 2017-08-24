import argparse
from time import strftime


def get_parser():
    parser = argparse.ArgumentParser(description='Parameters for DNN lab.')
    parser.add_argument(
        '--data-config',
        '--dc',
        type=str,
        required=True,
        help=
        'dataset config: dataFolderPath-firstTestFolder-nbTrainFolders-nbTestFolders, e.g., ../dataset/demoData/-3-2-1, or only dataFolder if using --prod, e.g., ../dataset/demoData/'
    )
    parser.add_argument(
        '--model-name',
        '-m',
        type=str,
        default=None,
        help='model name --lr | mlp | mlp_fe | elr | rf | xgb')
    parser.add_argument('--epochs', '-e', type=int, default=2, help='epochs')
    parser.add_argument(
        '--feed-mode',
        '--fm',
        type=str,
        default='batch',
        help='feed train data, can be batch | generator | all ')
    parser.add_argument(
        '--val-feed-mode',
        '--vfm',
        type=str,
        default='all',
        help='feed validation data, can be batch | all ')
    parser.add_argument(
        '--imb-learn',
        '--il',
        type=str,
        default='cw1',
        help=
        'imbalance compensation - os | us | cw | th (followed by value, e.g., os.33)'
    )
    parser.add_argument(
        '--id-prefix',
        '--ip',
        type=str,
        default=strftime('%m%d%H%M%S'),
        help='id for this launch (name prefix for all result records)')
    parser.add_argument(
        '--label',
        '-l',
        type=str,
        default='click',
        help='name of the output column (target)')
    parser.add_argument(
        '--no-cache', '--nc', action='store_true', help='not cache data')
    parser.add_argument(
        '--load-cache', '-c', action='store_true', help='load cached datasets')
    parser.add_argument(
        '--data-format',
        '--fmt',
        type=str,
        default='parquet',
        help='can be parquet | csv ')
    parser.add_argument(
        '--eval-trained',
        '--et',
        action='store_true',
        help='evaluate trained model')
    parser.add_argument(
        '--gen-load-threshold',
        '--glt',
        type=int,
        default=1000000,
        help='threshold for load generator')
    parser.add_argument(
        '--column-emb',
        '--ce',
        action='store_true',
        help='use individual embedding layer on each feature')
    parser.add_argument(
        '--max-feature',
        '--mf',
        type=str,
        default='1e7',
        help='max feature in Embedding layer')
    parser.add_argument(
        '--batch-size',
        '--bs',
        type=int,
        default=4096,
        help='batch size during training')
    parser.add_argument(
        '--steps-per-epoch',
        '--spe',
        type=int,
        default=4096,
        help=
        'steps per epoch in fit_generator, only relevant when feed_mode is generator'
    )
    parser.add_argument(
        '--trained-path',
        '--tp',
        type=str,
        default='results/trained_models/last_model.h5',
        help='path to store or load trained model')
    parser.add_argument(
        '--no-save',
        '--ns',
        action='store_true',
        help='don\'t save model and results when training ends')
    parser.add_argument(
        '--summary',
        '-s',
        action='store_true',
        help='show summary of model structure')
    parser.add_argument(
        '--continue-train',
        '--ct',
        action='store_true',
        help='continue training last model')
    parser.add_argument(
        '--no-info',
        '--ni',
        action='store_true',
        help=
        'not to count data statistics to avoid memory insufficiency (temporary)'
    )
    parser.add_argument(
        '--online-learning',
        '--ol',
        action='store_true',
        help='use online learning')
    parser.add_argument(
        '--verbose',
        '--v',
        type=int,
        default=1,
        help='verbose level, can be 0 | 1 | 2')
    parser.add_argument(
        '--csv-sep',
        '--cs',
        type=str,
        default=',',
        help='csv separator when using csv as data format')
    parser.add_argument(
        '--machine-learning',
        '--ml',
        type=str,
        default=None,
        help=
        'machine learning models, can be xgb (xgboost) | rf (randomForest) | adaboost | xgb-rf-adaboost'
    )

    parser.add_argument(
        '--export-option',
        '--eo',
        action='store_true',
        help='whether prompt the option to export model for tensorflow'
    )    
    parser.add_argument(
        '--no-eval',
        '--ne',
        action='store_true',
        help='skip evaluation'
    )

    parser.add_argument(
        '--predict-path',
        '--pp',
        type=str,
        help='file path of prediction data')

    parser.add_argument(
        '--prod', '-p', action='store_true', help='production mode')



    return parser


def print_args(args):
    print('\n\n========================\nArguments:')
    print(vars(args))
