def set_splitG3b(Param_dict):
    """
        Generate list of parameters.
        # TODO: More details to be added to documentation.
    :param Param_dict:
    :return: G3b_lst_param_eta
    """
    G3b_lst = Param_dict.get('G3b_lst')
    G3b_eta = Param_dict.get('G3b_eta')
    nG3b_eta = len(G3b_eta)

    G3b_lst_param_eta = [[] for _ in range(len(G3b_lst))]

    for etID1 in range(nG3b_eta):
        for etID2 in range(etID1 + 1):
            for etID3 in range(etID2 + 1):
                G3b_lst_param_eta[0].append([etID1, etID2, etID3])

    for etID1 in range(nG3b_eta):
        for etID2 in range(nG3b_eta):
            for etID3 in range(etID2 + 1):
                G3b_lst_param_eta[1].append([etID1, etID2, etID3])

    for etID1 in range(nG3b_eta):
        for etID2 in range(etID1 + 1):
            for etID3 in range(nG3b_eta):
                G3b_lst_param_eta[2].append([etID1, etID2, etID3])

    return G3b_lst_param_eta
