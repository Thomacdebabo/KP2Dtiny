KP2D128 = {'model_params': (32, 64, 128, 128, 128),
'nfeatures': 128,
'downsample': 3,
'top_k':150,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':128
}
KP2D88 = {'model_params': (16, 32, 32, 64, 64),
'nfeatures': 32,
'downsample': 2,
'top_k':150,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(88,88)
}
KP2D88sub = {'model_params': (16, 32, 32, 64, 64),
'nfeatures': 32,
'downsample': 2,
'top_k':150,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(88,88)

}
KP2D256 = {'model_params': (32, 64, 128, 128, 128),
'nfeatures': 128,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape': (256,256)
}
KP2D640 = {'model_params': (32, 64, 128, 128, 128),
'nfeatures': 128,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(480,640)
}
KP2D320 = {'model_params': (32, 64, 128, 128, 128),
'nfeatures': 128,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(240,320)
}
KP2Dv4 = {'model_params': (32, 64, 128, 256, 256),
'nfeatures': 256,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': True,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape':(240,320)
}
KP2Dv4Baseline = {'model_params': (32, 64, 128, 256, 256),
'nfeatures': 256,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': True,
'do_upsample': True,
'use_leaky_relu': True,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(240,320)
}

#Models for paper
## Baseline
KP2D88Baseline = {'model_params': (32, 64, 128, 256, 256),
'nfeatures': 256,
'downsample': 3,
'top_k':50,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': True,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(88,88)
}
KP2D320Baseline = {'model_params': (32, 64, 128, 256, 256),
'nfeatures': 256,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': True,
'do_upsample': True,
'use_leaky_relu': True,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(240,320)
}
KP2D640Baseline = {'model_params': (32, 64, 128, 256, 256),
'nfeatures': 256,
'downsample': 3,
'top_k':1000,
'do_cross': True,
'with_drop': True,
'do_upsample': True,
'use_leaky_relu': True,
'use_subpixel': True,
'with_io':True,
'legacy':True,
'shape':(480,640)
}

# Tiny V0
KP2D88tinyV0 = {'model_params': (16, 16, 32, 32, 32, 32), # use batch size 4
'nfeatures': 32,
'downsample': 1,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'large_feat': False,
'legacy':False,
'shape':(88,88)
}

KP2D320tinyV0 = {'model_params': (16, 16, 32, 32, 32, 32), # use batch size 4
'nfeatures': 32,
'downsample': 1,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'legacy':False,
'shape':(240,320)
}

## Tiny V1
KP2D88tinyV1 = {'model_params': (16, 32, 32, 64, 64, 32),
'nfeatures': 32,
'downsample': 2,
'top_k':150,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io': True,
'shape': (88, 88)
}

KP2D320tinyV1 = {'model_params': (16, 32, 32, 64, 64, 32),
'nfeatures': 32,
'downsample': 2,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape': (240, 320)
}
KP2D640tinyV1 = {'model_params': (16, 32, 32, 64, 64, 32),
'nfeatures': 32,
'downsample': 2,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape': (480, 640)
}

## Tiny V2
KP2D88tinyV2 = {'model_params': (16, 32, 64, 128, 128, 128),
'nfeatures': 64,
'downsample': 3,
'top_k':50,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape': (88, 88)
}
KP2D320tinyV2 = {'model_params': (16, 32, 64, 128, 128, 128),
'nfeatures': 64,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape': (240, 320)
}
KP2D640tinyV2 = {'model_params': (16, 32, 64, 128, 128, 128),
'nfeatures': 64,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape': (480, 640)
}

## Tiny V3
KP2D88tinyV3 = {'model_params': (16, 32, 64, 64, 128, 64),
'nfeatures': 128,
'downsample': 3,
'top_k':50,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape':(88,88)
}
KP2D320tinyV3 = {'model_params': (16, 32, 64, 64, 128, 64),
'nfeatures': 128,
'downsample': 3,
'top_k':300,
'do_cross': True,
'with_drop': False,
'do_upsample': True,
'use_leaky_relu': False,
'use_subpixel': True,
'with_io':True,
'shape':(240,320)
}

KP2D88tinyNano = {'model_params': (4, 8, 16, 16, 16),
'nfeatures': 16,
'downsample': 0,
'top_k':300,
'do_cross': True,
'with_drop': False,
'use_leaky_relu': False,
'do_upsample': False,
'with_io':True,
'shape':(88,88)
}

KP2D320tinyNano = {'model_params': (4, 8, 16, 16, 16),
'nfeatures': 16,
'downsample': 0,
'top_k':300,
'do_cross': True,
'with_drop': False,
'use_leaky_relu': False,
'do_upsample': False,
'with_io':True,
'shape': (240, 320)
}

