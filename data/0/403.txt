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
layer {
  name: "inception2-1-reduce"
  type: "Convolution"
  bottom: "inception2-1"
  top: "inception2-1-reduce"
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
    pad: 0
    kernel_size: 1
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
layer {
  name: "inception1-reduce"
  type: "Convolution"
  bottom: "inception1"
  top: "inception1-reduce"
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
    pad: 0
    kernel_size: 1
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
layer {
  name: "inception-scale-1"
  type: "Concat"
  bottom: "inception2-1-reduce"
  bottom: "inception1-reduce"
  top: "inception-scale-1"
}
layer {
  name: "relu-inception-scale-1"
  type: "ReLU"
  bottom: "inception-scale-1"
  top: "inception-scale-1"
}
layer {
  name: "inception2-1-resize"
  type: "Resize"
  bottom: "inception-scale-1"
  top: "inception2-1-resize"
  resize_param {
    is_pyramid_test: true
    out_height_scale: 2
    out_width_scale: 2
  }
}
layer {
  name: "conv-fc-scale-1_1"
  type: "Convolution"
  bottom: "inception2-1-resize"
  top: "conv-fc-scale-1_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
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
layer {
  name: "conv-out-scale-1"
  type: "Convolution"
  bottom: "conv-fc-scale-1_1"
  top: "conv-out-scale-1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 9
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
layer {
  name: "slice-scale-1"
  type: "Slice"
  bottom: "conv-out-scale-1"
  top: "score-scale-1"
  top: "bbox-lt-scale-1"
  top: "bbox-rt-scale-1"
  top: "bbox-rb-scale-1"
  top: "degree-scale-1"
  slice_param {
    slice_point: 1
    slice_point: 3
    slice_point: 5
    slice_point: 7
  }
}
layer {
  name: "inception2-2-reduce"
  type: "Convolution"
  bottom: "inception2-2"
  top: "inception2-2-reduce"
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
    pad: 0
    kernel_size: 1
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
layer {
  name: "inception-scale-2"
  type: "Concat"
  bottom: "inception2-2-reduce"
  bottom: "inception1-reduce"
  top: "inception-scale-2"
}
layer {
  name: "relu-inception-scale-2"
  type: "ReLU"
  bottom: "inception-scale-2"
  top: "inception-scale-2"
}
layer {
  name: "inception2-2-resize"
  type: "Resize"
  bottom: "inception-scale-2"
  top: "inception2-2-resize"
  resize_param {
    is_pyramid_test: true
    out_height_scale: 2
    out_width_scale: 2
  }
}
layer {
  name: "conv-fc-scale-2_1"
  type: "Convolution"
  bottom: "inception2-2-resize"
  top: "conv-fc-scale-2_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
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
layer {
  name: "conv-out-scale-2"
  type: "Convolution"
  bottom: "conv-fc-scale-2_1"
  top: "conv-out-scale-2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 9
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
layer {
  name: "slice-scale-2"
  type: "Slice"
  bottom: "conv-out-scale-2"
  top: "score-scale-2"
  top: "bbox-lt-scale-2"
  top: "bbox-rt-scale-2"
  top: "bbox-rb-scale-2"
  top: "degree-scale-2"
  slice_param {
    slice_point: 1
    slice_point: 3
    slice_point: 5
    slice_point: 7
  }
}
layer {
  name: "inception2-3-reduce"
  type: "Convolution"
  bottom: "inception2-3"
  top: "inception2-3-reduce"
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
    pad: 0
    kernel_size: 1
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
layer {
  name: "inception-scale-3"
  type: "Concat"
  bottom: "inception2-3-reduce"
  bottom: "inception1-reduce"
  top: "inception-scale-3"
}
layer {
  name: "relu-inception-scale-3"
  type: "ReLU"
  bottom: "inception-scale-3"
  top: "inception-scale-3"
}
layer {
  name: "inception2-3-resize"
  type: "Resize"
  bottom: "inception-scale-3"
  top: "inception2-3-resize"
  resize_param {
    is_pyramid_test: true
    out_height_scale: 2
    out_width_scale: 2
  }
}
layer {
  name: "conv-fc-scale-3_1"
  type: "Convolution"
  bottom: "inception2-3-resize"
  top: "conv-fc-scale-3_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
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
layer {
  name: "conv-out-scale-3"
  type: "Convolution"
  bottom: "conv-fc-scale-3_1"
  top: "conv-out-scale-3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 9
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
layer {
  name: "slice-scale-3"
  type: "Slice"
  bottom: "conv-out-scale-3"
  top: "score-scale-3"
  top: "bbox-lt-scale-3"
  top: "bbox-rt-scale-3"
  top: "bbox-rb-scale-3"
  top: "degree-scale-3"
  slice_param {
    slice_point: 1
    slice_point: 3
    slice_point: 5
    slice_point: 7
  }
}
layer {
  name: "concat_all"
  type: "Concat"
  bottom: "score-scale-1"
  bottom: "bbox-lt-scale-1"
  bottom: "bbox-rb-scale-1"
  bottom: "score-scale-2"
  bottom: "bbox-lt-scale-2"
  bottom: "bbox-rb-scale-2"
  bottom: "score-scale-3"
  bottom: "bbox-lt-scale-3"
  bottom: "bbox-rb-scale-3"
  top: "res_all"
}
