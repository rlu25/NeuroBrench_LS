def create_model(opt):
    if opt.model == 'local':
        from .PTN_model import PTN_local
        model = PTN_local()
    elif opt.model == 'vanilla':
        from .PTN_model import PTN
        model = PTN()
    elif opt.model == '3layers':
        from .PTN_model import PTN_local3
        model = PTN_local3()
    elif opt.model == '4layers':
        from .PTN_model import PTN_local4
        model = PTN_local4()
    elif opt.model == 'local3D':
        from .PTN_model3D import PTN_local
        model = PTN_local()
    elif opt.model == 'general3D':
        from .PTN_model3D import PTN
        model = PTN()
    elif opt.model == 'PTN_shallowU':
        from .PTN_model import PTN_shallowU
        model  = PTN_shallowU()
    elif opt.model == 'Local_ps':
        from .PTN_model_pixelshuffle import PTN_local
        model  = PTN_local()
    elif opt.model == 'Local_ps3D':
        from .PTN_model3D_ps import PTN_local
        model  = PTN_local()
    elif opt.model =='local3D_V2':
        from .PTN_model3D import PTNet_local2
        model = PTNet_local2()
    elif opt.model =='local3D_trans':
        from .PTN_model3D import PTN_local_trans
        model = PTN_local_trans()
    elif opt.model =='local3D_trans3layer':
        from .PTN_model3D import PTN_local_trans2
        model = PTN_local_trans2()
    elif opt.model =='local2D_trans':
        from .PTNModel2D_latest import PTN_Local_Trans
        model = PTN_Local_Trans()
    elif opt.model =='local3D_transnoSC':
        from .PTN_model3D import PTN_local_trans_noSC
        model = PTN_local_trans_noSC()
    return model
