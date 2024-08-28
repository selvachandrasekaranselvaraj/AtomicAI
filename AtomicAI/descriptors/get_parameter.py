from AtomicAI.descriptors.set_parameter import set_param_dict
def get_parameters():
    parameters = {
    #      cut off for fingerprint
    #        AA
    'Rc2b': 1.5 * 5.0,
    'Rc3b': 1.5 * 5.0,
    'Reta': 1.5 * 5.0,
    #        |    2-body term      |
    #        |    Eta       |  Rs  |
    #        min   max   num| dRs  |
    #        AA    AA    int  AA
    '2b': [-3.0, 1.0, 20, 2.5],
    #      |  3-body term |
    #      |        Eta   | Rs  | zeta | theta |
    #      min   max   num| dRs | num  |  num  |
    #      AA    AA    int|  AA | int  |  int  |
    '3b': [-3.0, 1.0, 20, 10.5, 3, 10],
    #        |split 3-body term|
    #        | min   max   num|
    #        | AA    AA    int|
    'split3b': [-3.0, 1.0, 10]
        }
    fp_flag = 'Split2b3b'
    param_dict = set_param_dict(parameters, fp_flag)
    return param_dict
