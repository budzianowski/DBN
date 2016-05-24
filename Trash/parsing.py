import sys
import copy

def get_arg(arg, args, default, type_):
    arg = '--'+arg
    if arg in args :
        index = args.index(arg)
        value = args[args.index(arg) + 1]
        del args[index]     #remove arg-name
        del args[index]     #remove value
        return type_(value)
    else:
        return default


def get_flag(flag, args):
    flag = '--'+flag
    have_flag = flag in args
    if have_flag :
        args.remove(flag)

    return have_flag

def parse_args(command_line_args, command_line_flags):
    args = copy.deepcopy(sys.argv[1:])
    arg_dict = {}
    for (arg_name, arg_args) in command_line_args.items():
        (arg_default_val, arg_type) = arg_args
        arg_dict[arg_name] = get_arg(arg_name, args, arg_default_val, arg_type)

    for flag_name in command_line_flags:
        arg_dict[flag_name] = get_flag(flag_name, args)

    if len(args) > 0:
        print('Have unused args: {0}'.format(args))

    return arg_dict


def print_args(args):
    print('Parameters used:')
    print('--------------------------------------')
    for (k, v) in args.items():
        print('\t{0}: {1}'.format(k, v))
    print('--------------------------------------')
