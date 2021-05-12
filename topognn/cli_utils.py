import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def int_or_none(input: str):
    try:
        return int(input)
    except Exception as e:
        if input.lower() == 'none':
            return None
        else:
            argparse.ArgumentTypeError(f'"{input}" is neither None nor of integer type.')
