import argparse
import configargparse
import os

def print_options(self, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'DisguisOR'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='DisguisOR')

    # general settings                              
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--experiment', type=str, default="hard", choices=["hard"] ,help='name of the experiment.')
    
    # Registration related
    parser.add_argument('--voxel_size', type=float, default=0.0125, help='voxel size for registration')
    parser.add_argument('--max_iteration_icp', type=int, default=250, help='max iteration for icp')
    parser.add_argument('--max_iteration_filterreg', type=int, default=1825000000, help='max iteration for filterreg')
    parser.add_argument('--filterreg_w', type=float, default=0.0001, help='weight for filterreg')
    parser.add_argument('--filterreg_sigma2', type=float, default=0.000435, help='sigma2 for filterreg')
    parser.add_argument('--filterreg_tol', type=float, default=0.00005, help='tolerance for filterreg')

    # Render related
    parser.add_argument('--alpha_value', type=float, default=0.725, help='alpha value for poisson image editing')
    parser.add_argument('--add_bboxes', action='store_true', help='add bounding boxes to the rendered images')

    parser.add_argument('--texture_list', nargs='+', default=['input/texture/1.jpg', 'input/texture/2.jpg'], type=str, help='texture list for rendering')
    parser.add_argument('--default_texture', type=str, default='input/texture/1.jpg', help='default texture for rendering')

    args, _ = parser.parse_known_args()

    return args