import yaml

class ParamStruct(object):
    def __init__(self,args):
        self.__dict__.update(**args)


def read_yaml(path):
    '''
    Read yaml path and return python object
    '''
    with open(path,'r') as f:
        return yaml.load(f.read())

def build_param_struct(args):
    '''
    Build a param struct of depth 2
    '''
    params = ParamStruct(args)
    for key in args.keys():
        params.__setattr__(key,ParamStruct(params.__getattribute__(key)))
    return params

def read_params(path):
    ''' Read yaml and build param struct'''
    return build_param_struct(read_yaml(path))
