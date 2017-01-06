input:"data"
input_dim:1
input_dim:3
input_dim:64
input_dim:64

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 3
    kernel_size: 7
    stride: 4
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}

layer {
  name: "inception1_3x3_1_conv"
  type: "Convolution"
  bottom: "conv1"
  top: "inception1_3x3_1_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 0
    pad_w: 0
    stride_h: 1
    stride_w: 1
  }
}
layer {
  name: "inception1_3x3_1_relu"
  type: "ReLU"
  bottom: "inception1_3x3_1_conv"
  top: "inception1_3x3_1_conv"
}
layer {
  name: "inception1_3x3_2_conv"
  type: "Convolution"
  bottom: "inception1_3x3_1_conv"
  top: "inception1_3x3_2_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 72
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 1
    pad_w: 1
  }
}

layer {
  name: "inception1_5x5_1_conv"
  type: "Convolution"
  bottom: "conv1"
  top: "inception1_5x5_1_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 0
    pad_w: 0
    stride_h: 1
    stride_w: 1
  }
}
layer {
  name: "inception1_5x5_1_relu"
  type: "ReLU"
  bottom: "inception1_5x5_1_conv"
  top: "inception1_5x5_1_conv"
}
layer {
  name: "inception1_5x5_2_conv"
  type: "Convolution"
  bottom: "inception1_5x5_1_conv"
  top: "inception1_5x5_2_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 32
    kernel_size: 5
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 2
    pad_w: 2
  }
}
layer {
  name: "inception1_concat"
  type: "Concat"
  bottom: "inception1_3x3_2_conv"
  bottom: "inception1_5x5_2_conv"
  top: "inception1"
}
layer {
  name: "inception1_relu"
  type: "ReLU"
  bottom: "inception1"
  top: "inception1"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "inception1"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}

layer {
  name: "inception2-1_3x3_1_conv"
  type: "Convolution"
  bottom: "pool2"
  top: "inception2-1_3x3_1_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 0
    pad_w: 0
    stride_h: 1
    stride_w: 1
  }
}
layer {
  name: "inception2-1_3x3_1_relu"
  type: "ReLU"
  bottom: "inception2-1_3x3_1_conv"
  top: "inception2-1_3x3_1_conv"
}
layer {
  name: "inception2-1_3x3_2_conv"
  type: "Convolution"
  bottom: "inception2-1_3x3_1_conv"
  top: "inception2-1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    group: 2
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
  }
}
layer {
  name: "inception2-1_relu"
  type: "ReLU"
  bottom: "inception2-1"
  top: "inception2-1"
}
layer {
  name: "inception2-2_3x3_1_conv"
  type: "Convolution"
  bottom: "inception2-1"
  top: "inception2-2_3x3_1_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 0
    pad_w: 0
    stride_h: 1
    stride_w: 1
  }
}
layer {
  name: "inception2-2_3x3_1_relu"
  type: "ReLU"
  bottom: "inception2-2_3x3_1_conv"
  top: "inception2-2_3x3_1_conv"
}
layer {
  name: "inception2-2_3x3_2_conv"
  type: "Convolution"
  bottom: "inception2-2_3x3_1_conv"
  top: "inception2-2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    group: 2
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
  }
}
layer {
  name: "inception2-2_relu"
  type: "ReLU"
  bottom: "inception2-2"
  top: "inception2-2"
}

layer {
  name: "inception2-3_3x3_1_conv"
  type: "Convolution"
  bottom: "inception2-2"
  top: "inception2-3_3x3_1_conv"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 0
    pad_w: 0
    stride_h: 1
    stride_w: 1
  }
}
layer {
  name: "inception2-3_3x3_1_relu"
  type: "ReLU"
  bottom: "inception2-3_3x3_1_conv"
  top: "inception2-3_3x3_1_conv"
}
layer {
  name: "inception2-3_3x3_2_conv"
  type: "Convolution"
  bottom: "inception2-3_3x3_1_conv"
  top: "inception2-3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    group: 2
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
  }
}
layer {
  name: "inception2-3_relu"
  type: "ReLU"
  bottom: "inception2-3"
  top: "inception2-3"
}

########### scale_base: 0.6  use inception2-1 #########

layer{
  name: "inception2-1-reduce"
  type: "Convolution"
  bottom: "inception2-1"
  top: "inception2-1-reduce"
  param{
    lr_mult: 1
    decay_mult: 1
  }
  param{
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 1
    pad: 0
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
     }
  }
}

layer{
  name: "inception1-reduce"
  type: "Convolution"
  bottom: "inception1"
  top: "inception1-reduce"
  param{
    	lr_mult: 1
  	decay_mult: 1
  }
  param{
    	lr_mult: 2
  	decay_mult: 0
  }
  convolution_param {
      num_output: 64
      kernel_size: 1
      pad: 0
      stride: 1
      weight_filler {
        type: "xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
  }
}

layer{
  name: "inception-scale-1"
  type: "Concat"
  bottom: "inception2-1-reduce"
  bottom: "inception1-reduce"
  top: "inception-scale-1"
}
layer { 
  bottom: "inception-scale-1"
  top: "inception-scale-1"
  name: "relu-inception-scale-1"
  type: "ReLU"
}

layer{
  name: "inception2-1-resize"	
  type: "Resize"
  bottom: "inception-scale-1"
  top: "inception2-1-resize"
  resize_param{
    is_pyramid_test: true
    out_height_scale: 2
    out_width_scale: 2
  }
}

layer{
  name: "conv-fc-scale-1"
  type: "Convolution"
  bottom: "inception2-1-resize"
  top: "conv-fc-scale-1"
    param{
    	lr_mult: 1
  	decay_mult: 1
    }
    param{
    	lr_mult: 2
  	decay_mult: 0
    }
    convolution_param {
      num_output: 128
      kernel_size: 1
      weight_filler {
        type: "xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}

layer{
  name: "conv-out-scale-1"
  type: "Convolution"
  bottom: "conv-fc-scale-1"
  top: "conv-out-scale-1"
    param{
    	lr_mult: 1
  	decay_mult: 1
    }
    param{
    	lr_mult: 2
  	decay_mult: 0
    }
    convolution_param {
      num_output: 5
      kernel_size: 1
      weight_filler {
        type: "xavier"
        # type: "gaussian"
        # std: 0.01
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}
########### scale_base: 0.84 use inception2-2 #########

layer{
  name: "inception2-2-reduce"
  type: "Convolution"
  bottom: "inception2-2"
  top: "inception2-2-reduce"
    param{
    	lr_mult: 1
  	decay_mult: 1
    }
    param{
    	lr_mult: 2
  	decay_mult: 0
    }
    convolution_param {
      num_output: 64
      kernel_size: 1
	  pad: 0
      stride: 1
      weight_filler {
        type: "xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}

layer{
  name: "inception-scale-2"
  type: "Concat"
  bottom: "inception2-2-reduce"
  bottom: "inception1-reduce"
  top: "inception-scale-2"
}

layer { 
  bottom: "inception-scale-2"
  top: "inception-scale-2"
  name: "relu-inception-scale-2"
  type: "ReLU"
}

layer{
	type: "Resize"
	name: "inception2-2-resize"
	bottom: "inception-scale-2"
	top: "inception2-2-resize"
	resize_param{
		is_pyramid_test: true
		out_height_scale: 2
		out_width_scale: 2
	}
}

layer{
	type: "Convolution"
	name: "conv-fc-scale-2"
	bottom: "inception2-2-resize"
	top: "conv-fc-scale-2"
    param{
    	lr_mult: 1
  		decay_mult: 1
    }
    param{
    	lr_mult: 2
  		decay_mult: 0
    }
    convolution_param {
      num_output: 128
      kernel_size: 1
      weight_filler {
        type: "xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}
 
layer{
	type: "Convolution"
	name: "conv-out-scale-2"
	bottom: "conv-fc-scale-2"
	top: "conv-out-scale-2"
    param{
    	lr_mult: 1
  		decay_mult: 1
    }
    param{
    	lr_mult: 2
  		decay_mult: 0
    }
    convolution_param {
      num_output: 5
      kernel_size: 1
      weight_filler {
        type: "xavier"
        # type: "gaussian"
        # std: 0.01
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}

########### scale_base: 1 use inception2-3 #########

layer{
	name: "inception2-3-reduce"
	type: "Convolution"
	bottom: "inception2-3"
	top: "inception2-3-reduce"
    param{
    	lr_mult: 1
  		decay_mult: 1
    }
    param{
    	lr_mult: 2
  		decay_mult: 0
    }
    convolution_param {
      num_output: 64
      kernel_size: 1
	  pad: 0
      stride: 1
      weight_filler {
        type: "xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}

layer{
	name: "inception-scale-3"
	type: "Concat"
	bottom: "inception2-3-reduce"
	bottom: "inception1-reduce"
	top: "inception-scale-3"
}

layer { 
  bottom: "inception-scale-3"
  top: "inception-scale-3"
  name: "relu-inception-scale-3"
  type: "ReLU"
}

layer{
	type: "Resize"
	name: "inception2-3-resize"
	bottom: "inception-scale-3"
	top: "inception2-3-resize"
	resize_param{
		is_pyramid_test: true
		out_height_scale: 2
		out_width_scale: 2
	}
}

layer{
	type: "Convolution"
	name: "conv-fc-scale-3"
	bottom: "inception2-3-resize"
	top: "conv-fc-scale-3"
    param{
    	lr_mult: 1
  		decay_mult: 1
    }
    param{
    	lr_mult: 2
  		decay_mult: 0
    }
    convolution_param {
      num_output: 128
      kernel_size: 1
      weight_filler {
        type: "xavier"
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}

layer{
	type: "Convolution"
	name: "conv-out-scale-3"
	bottom: "conv-fc-scale-3"
	top: "conv-out-scale-3"
    param{
    	lr_mult: 1
  		decay_mult: 1
    }
    param{
    	lr_mult: 2
  		decay_mult: 0
    }
    convolution_param {
      num_output: 5
      kernel_size: 1
      weight_filler {
        type: "xavier"
        # type: "gaussian"
        # std: 0.01
      }
      bias_filler {
        type: "constant"
        value: 0
      }
    }
}

layer{
	type: "Concat"
	name: "concat_all"
	bottom: "conv-out-scale-1"
	bottom: "conv-out-scale-2"
	bottom: "conv-out-scale-3"

	top: "res_all"
}


