input:"face_patch_data" 
input_dim:1
input_dim:21
input_dim:108
input_dim:108
layer {
  name: "slice_data"
  type: "Slice"
  bottom: "face_patch_data"
  top: "patch1"
  top: "patch2"
  top: "patch3"
  top: "patch4"
  top: "patch5"
  top: "patch6"
  top: "patch0"
  slice_param {
    slice_dim: 1
    #slice_point: 3
  }
}
# start patch_1 network!
layer {
  name: "patch_1_conv0"
  type: "Convolution"
  bottom: "patch1"
  top: "patch_1_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv0_w"
 #param: "patch_1_conv0_b"
}
layer {
  name: "batch_nor_patch_1_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv0"
  top: "patch_1_conv0"
  
  
  
}
layer {
  name: "relu_patch_1_conv0"
  type: "ReLU"
  bottom: "patch_1_conv0"
  top: "patch_1_conv0"
}
layer {
  name: "patch_1_conv1-1_1"
  type: "Convolution"
  bottom: "patch_1_conv0"
  top: "patch_1_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv1-1_1_w"
 #param: "patch_1_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_1_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv1-1_1"
  top: "patch_1_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv1-1_1"
  type: "ReLU"
  bottom: "patch_1_conv1-1_1"
  top: "patch_1_conv1-1_1"
}
layer {
  name: "patch_1_conv1-1_2"
  type: "Convolution"
  bottom: "patch_1_conv1-1_1"
  top: "patch_1_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv1-1_2_w"
 #param: "patch_1_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_1_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv1-1_2"
  top: "patch_1_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv1-1_2"
  type: "ReLU"
  bottom: "patch_1_conv1-1_2"
  top: "patch_1_conv1-1_2"
}
layer {
  name: "patch_1_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_1_conv1-1_2"
  bottom:"patch_1_conv0"
  top: "patch_1_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv1-2_1"
  type: "Convolution"
  bottom: "patch_1_shortcut1-1"
  top: "patch_1_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv1-2_1_w"
 #param: "patch_1_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_1_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv1-2_1"
  top: "patch_1_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv1-2_1"
  type: "ReLU"
  bottom: "patch_1_conv1-2_1"
  top: "patch_1_conv1-2_1"
}
layer {
  name: "patch_1_conv1-2_2"
  type: "Convolution"
  bottom: "patch_1_conv1-2_1"
  top: "patch_1_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv1-2_2_w"
 #param: "patch_1_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_1_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv1-2_2"
  top: "patch_1_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv1-2_2"
  type: "ReLU"
  bottom: "patch_1_conv1-2_2"
  top: "patch_1_conv1-2_2"
}
layer {
  name: "patch_1_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_1_conv1-2_2"
  bottom:"patch_1_shortcut1-1"
  top: "patch_1_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv1-3_1"
  type: "Convolution"
  bottom: "patch_1_shortcut1-2"
  top: "patch_1_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv1-3_1_w"
 #param: "patch_1_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_1_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv1-3_1"
  top: "patch_1_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv1-3_1"
  type: "ReLU"
  bottom: "patch_1_conv1-3_1"
  top: "patch_1_conv1-3_1"
}
layer {
  name: "patch_1_conv1-3_2"
  type: "Convolution"
  bottom: "patch_1_conv1-3_1"
  top: "patch_1_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv1-3_2_w"
 #param: "patch_1_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_1_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv1-3_2"
  top: "patch_1_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv1-3_2"
  type: "ReLU"
  bottom: "patch_1_conv1-3_2"
  top: "patch_1_conv1-3_2"
}
layer {
  name: "patch_1_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_1_conv1-3_2"
  bottom:"patch_1_shortcut1-2"
  top: "patch_1_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv2-1_1"
  type: "Convolution"
  bottom: "patch_1_shortcut1-3"
  top: "patch_1_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-1_1_w"
 #param: "patch_1_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_1_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-1_1"
  top: "patch_1_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-1_1"
  type: "ReLU"
  bottom: "patch_1_conv2-1_1"
  top: "patch_1_conv2-1_1"
}
layer {
  name: "patch_1_conv2-1_2"
  type: "Convolution"
  bottom: "patch_1_conv2-1_1"
  top: "patch_1_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-1_2_w"
 #param: "patch_1_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_1_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-1_2"
  top: "patch_1_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-1_2"
  type: "ReLU"
  bottom: "patch_1_conv2-1_2"
  top: "patch_1_conv2-1_2"
}
layer {
  name: "patch_1_project1"
  type: "Convolution"
  bottom: "patch_1_shortcut1-3"
  top: "patch_1_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_project1_w"
 #param: "patch_1_project1_b"
}
layer {
  name: "batch_nor_patch_1_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_project1"
  top: "patch_1_project1"
  
  
  
}
layer {
  name: "relu_patch_1_project1"
  type: "ReLU"
  bottom: "patch_1_project1"
  top: "patch_1_project1"
}
layer {
  name: "patch_1_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_1_conv2-1_2"
  bottom:"patch_1_project1"
  top: "patch_1_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv2-2_1"
  type: "Convolution"
  bottom: "patch_1_shortcut2-1"
  top: "patch_1_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-2_1_w"
 #param: "patch_1_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_1_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-2_1"
  top: "patch_1_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-2_1"
  type: "ReLU"
  bottom: "patch_1_conv2-2_1"
  top: "patch_1_conv2-2_1"
}
layer {
  name: "patch_1_conv2-2_2"
  type: "Convolution"
  bottom: "patch_1_conv2-2_1"
  top: "patch_1_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-2_2_w"
 #param: "patch_1_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_1_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-2_2"
  top: "patch_1_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-2_2"
  type: "ReLU"
  bottom: "patch_1_conv2-2_2"
  top: "patch_1_conv2-2_2"
}
layer {
  name: "patch_1_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_1_conv2-2_2"
  bottom:"patch_1_shortcut2-1"
  top: "patch_1_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv2-3_1"
  type: "Convolution"
  bottom: "patch_1_shortcut2-2"
  top: "patch_1_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-3_1_w"
 #param: "patch_1_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_1_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-3_1"
  top: "patch_1_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-3_1"
  type: "ReLU"
  bottom: "patch_1_conv2-3_1"
  top: "patch_1_conv2-3_1"
}
layer {
  name: "patch_1_conv2-3_2"
  type: "Convolution"
  bottom: "patch_1_conv2-3_1"
  top: "patch_1_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-3_2_w"
 #param: "patch_1_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_1_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-3_2"
  top: "patch_1_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-3_2"
  type: "ReLU"
  bottom: "patch_1_conv2-3_2"
  top: "patch_1_conv2-3_2"
}
layer {
  name: "patch_1_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_1_conv2-3_2"
  bottom:"patch_1_shortcut2-2"
  top: "patch_1_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv2-4_1"
  type: "Convolution"
  bottom: "patch_1_shortcut2-3"
  top: "patch_1_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-4_1_w"
 #param: "patch_1_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_1_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-4_1"
  top: "patch_1_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-4_1"
  type: "ReLU"
  bottom: "patch_1_conv2-4_1"
  top: "patch_1_conv2-4_1"
}
layer {
  name: "patch_1_conv2-4_2"
  type: "Convolution"
  bottom: "patch_1_conv2-4_1"
  top: "patch_1_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv2-4_2_w"
 #param: "patch_1_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_1_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv2-4_2"
  top: "patch_1_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv2-4_2"
  type: "ReLU"
  bottom: "patch_1_conv2-4_2"
  top: "patch_1_conv2-4_2"
}
layer {
  name: "patch_1_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_1_conv2-4_2"
  bottom:"patch_1_shortcut2-3"
  top: "patch_1_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv3-1_1"
  type: "Convolution"
  bottom: "patch_1_shortcut2-4"
  top: "patch_1_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-1_1_w"
 #param: "patch_1_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_1_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-1_1"
  top: "patch_1_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-1_1"
  type: "ReLU"
  bottom: "patch_1_conv3-1_1"
  top: "patch_1_conv3-1_1"
}
layer {
  name: "patch_1_conv3-1_2"
  type: "Convolution"
  bottom: "patch_1_conv3-1_1"
  top: "patch_1_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-1_2_w"
 #param: "patch_1_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_1_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-1_2"
  top: "patch_1_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-1_2"
  type: "ReLU"
  bottom: "patch_1_conv3-1_2"
  top: "patch_1_conv3-1_2"
}
layer {
  name: "patch_1_project2"
  type: "Convolution"
  bottom: "patch_1_shortcut2-4"
  top: "patch_1_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_project2_w"
 #param: "patch_1_project2_b"
}
layer {
  name: "batch_nor_patch_1_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_project2"
  top: "patch_1_project2"
  
  
  
}
layer {
  name: "relu_patch_1_project2"
  type: "ReLU"
  bottom: "patch_1_project2"
  top: "patch_1_project2"
}
layer {
  name: "patch_1_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_1_conv3-1_2"
  bottom:"patch_1_project2"
  top: "patch_1_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv3-2_1"
  type: "Convolution"
  bottom: "patch_1_shortcut3-1"
  top: "patch_1_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-2_1_w"
 #param: "patch_1_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_1_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-2_1"
  top: "patch_1_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-2_1"
  type: "ReLU"
  bottom: "patch_1_conv3-2_1"
  top: "patch_1_conv3-2_1"
}
layer {
  name: "patch_1_conv3-2_2"
  type: "Convolution"
  bottom: "patch_1_conv3-2_1"
  top: "patch_1_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-2_2_w"
 #param: "patch_1_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_1_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-2_2"
  top: "patch_1_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-2_2"
  type: "ReLU"
  bottom: "patch_1_conv3-2_2"
  top: "patch_1_conv3-2_2"
}
layer {
  name: "patch_1_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_1_conv3-2_2"
  bottom:"patch_1_shortcut3-1"
  top: "patch_1_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv3-3_1"
  type: "Convolution"
  bottom: "patch_1_shortcut3-2"
  top: "patch_1_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-3_1_w"
 #param: "patch_1_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_1_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-3_1"
  top: "patch_1_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-3_1"
  type: "ReLU"
  bottom: "patch_1_conv3-3_1"
  top: "patch_1_conv3-3_1"
}
layer {
  name: "patch_1_conv3-3_2"
  type: "Convolution"
  bottom: "patch_1_conv3-3_1"
  top: "patch_1_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-3_2_w"
 #param: "patch_1_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_1_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-3_2"
  top: "patch_1_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-3_2"
  type: "ReLU"
  bottom: "patch_1_conv3-3_2"
  top: "patch_1_conv3-3_2"
}
layer {
  name: "patch_1_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_1_conv3-3_2"
  bottom:"patch_1_shortcut3-2"
  top: "patch_1_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv3-4_1"
  type: "Convolution"
  bottom: "patch_1_shortcut3-3"
  top: "patch_1_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-4_1_w"
 #param: "patch_1_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_1_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-4_1"
  top: "patch_1_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-4_1"
  type: "ReLU"
  bottom: "patch_1_conv3-4_1"
  top: "patch_1_conv3-4_1"
}
layer {
  name: "patch_1_conv3-4_2"
  type: "Convolution"
  bottom: "patch_1_conv3-4_1"
  top: "patch_1_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-4_2_w"
 #param: "patch_1_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_1_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-4_2"
  top: "patch_1_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-4_2"
  type: "ReLU"
  bottom: "patch_1_conv3-4_2"
  top: "patch_1_conv3-4_2"
}
layer {
  name: "patch_1_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_1_conv3-4_2"
  bottom:"patch_1_shortcut3-3"
  top: "patch_1_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv3-5_1"
  type: "Convolution"
  bottom: "patch_1_shortcut3-4"
  top: "patch_1_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-5_1_w"
 #param: "patch_1_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_1_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-5_1"
  top: "patch_1_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-5_1"
  type: "ReLU"
  bottom: "patch_1_conv3-5_1"
  top: "patch_1_conv3-5_1"
}
layer {
  name: "patch_1_conv3-5_2"
  type: "Convolution"
  bottom: "patch_1_conv3-5_1"
  top: "patch_1_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-5_2_w"
 #param: "patch_1_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_1_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-5_2"
  top: "patch_1_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-5_2"
  type: "ReLU"
  bottom: "patch_1_conv3-5_2"
  top: "patch_1_conv3-5_2"
}
layer {
  name: "patch_1_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_1_conv3-5_2"
  bottom:"patch_1_shortcut3-4"
  top: "patch_1_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv3-6_1"
  type: "Convolution"
  bottom: "patch_1_shortcut3-5"
  top: "patch_1_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-6_1_w"
 #param: "patch_1_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_1_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-6_1"
  top: "patch_1_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-6_1"
  type: "ReLU"
  bottom: "patch_1_conv3-6_1"
  top: "patch_1_conv3-6_1"
}
layer {
  name: "patch_1_conv3-6_2"
  type: "Convolution"
  bottom: "patch_1_conv3-6_1"
  top: "patch_1_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv3-6_2_w"
 #param: "patch_1_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_1_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv3-6_2"
  top: "patch_1_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv3-6_2"
  type: "ReLU"
  bottom: "patch_1_conv3-6_2"
  top: "patch_1_conv3-6_2"
}
layer {
  name: "patch_1_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_1_conv3-6_2"
  bottom:"patch_1_shortcut3-5"
  top: "patch_1_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv4-1_1"
  type: "Convolution"
  bottom: "patch_1_shortcut3-6"
  top: "patch_1_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv4-1_1_w"
 #param: "patch_1_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_1_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv4-1_1"
  top: "patch_1_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv4-1_1"
  type: "ReLU"
  bottom: "patch_1_conv4-1_1"
  top: "patch_1_conv4-1_1"
}
layer {
  name: "patch_1_conv4-1_2"
  type: "Convolution"
  bottom: "patch_1_conv4-1_1"
  top: "patch_1_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv4-1_2_w"
 #param: "patch_1_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_1_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv4-1_2"
  top: "patch_1_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv4-1_2"
  type: "ReLU"
  bottom: "patch_1_conv4-1_2"
  top: "patch_1_conv4-1_2"
}
layer {
  name: "patch_1_project3"
  type: "Convolution"
  bottom: "patch_1_shortcut3-6"
  top: "patch_1_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_project3_w"
 #param: "patch_1_project3_b"
}
layer {
  name: "batch_nor_patch_1_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_project3"
  top: "patch_1_project3"
  
  
  
}
layer {
  name: "relu_patch_1_project3"
  type: "ReLU"
  bottom: "patch_1_project3"
  top: "patch_1_project3"
}
layer {
  name: "patch_1_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_1_conv4-1_2"
  bottom:"patch_1_project3"
  top: "patch_1_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv4-2_1"
  type: "Convolution"
  bottom: "patch_1_shortcut4-1"
  top: "patch_1_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv4-2_1_w"
 #param: "patch_1_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_1_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv4-2_1"
  top: "patch_1_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv4-2_1"
  type: "ReLU"
  bottom: "patch_1_conv4-2_1"
  top: "patch_1_conv4-2_1"
}
layer {
  name: "patch_1_conv4-2_2"
  type: "Convolution"
  bottom: "patch_1_conv4-2_1"
  top: "patch_1_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv4-2_2_w"
 #param: "patch_1_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_1_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv4-2_2"
  top: "patch_1_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv4-2_2"
  type: "ReLU"
  bottom: "patch_1_conv4-2_2"
  top: "patch_1_conv4-2_2"
}
layer {
  name: "patch_1_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_1_conv4-2_2"
  bottom:"patch_1_shortcut4-1"
  top: "patch_1_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_conv4-3_1"
  type: "Convolution"
  bottom: "patch_1_shortcut4-2"
  top: "patch_1_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv4-3_1_w"
 #param: "patch_1_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_1_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv4-3_1"
  top: "patch_1_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_1_conv4-3_1"
  type: "ReLU"
  bottom: "patch_1_conv4-3_1"
  top: "patch_1_conv4-3_1"
}
layer {
  name: "patch_1_conv4-3_2"
  type: "Convolution"
  bottom: "patch_1_conv4-3_1"
  top: "patch_1_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_conv4-3_2_w"
 #param: "patch_1_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_1_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_1_conv4-3_2"
  top: "patch_1_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_1_conv4-3_2"
  type: "ReLU"
  bottom: "patch_1_conv4-3_2"
  top: "patch_1_conv4-3_2"
}
layer {
  name: "patch_1_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_1_conv4-3_2"
  bottom:"patch_1_shortcut4-2"
  top: "patch_1_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_1_pool1"
  type: "Pooling"
  bottom: "patch_1_shortcut4-3"
  top: "patch_1_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_1_fc1"
  type: "InnerProduct"
  bottom: "patch_1_pool1"
  top: "patch_1_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_1_fc1_w"
 #param: "patch_1_fc1_b"
}

# start patch_2 network!
layer {
  name: "patch_2_conv0"
  type: "Convolution"
  bottom: "patch2"
  top: "patch_2_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv0_w"
 #param: "patch_2_conv0_b"
}
layer {
  name: "batch_nor_patch_2_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv0"
  top: "patch_2_conv0"
  
  
  
}
layer {
  name: "relu_patch_2_conv0"
  type: "ReLU"
  bottom: "patch_2_conv0"
  top: "patch_2_conv0"
}
layer {
  name: "patch_2_conv1-1_1"
  type: "Convolution"
  bottom: "patch_2_conv0"
  top: "patch_2_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv1-1_1_w"
 #param: "patch_2_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_2_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv1-1_1"
  top: "patch_2_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv1-1_1"
  type: "ReLU"
  bottom: "patch_2_conv1-1_1"
  top: "patch_2_conv1-1_1"
}
layer {
  name: "patch_2_conv1-1_2"
  type: "Convolution"
  bottom: "patch_2_conv1-1_1"
  top: "patch_2_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv1-1_2_w"
 #param: "patch_2_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_2_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv1-1_2"
  top: "patch_2_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv1-1_2"
  type: "ReLU"
  bottom: "patch_2_conv1-1_2"
  top: "patch_2_conv1-1_2"
}
layer {
  name: "patch_2_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_2_conv1-1_2"
  bottom:"patch_2_conv0"
  top: "patch_2_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv1-2_1"
  type: "Convolution"
  bottom: "patch_2_shortcut1-1"
  top: "patch_2_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv1-2_1_w"
 #param: "patch_2_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_2_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv1-2_1"
  top: "patch_2_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv1-2_1"
  type: "ReLU"
  bottom: "patch_2_conv1-2_1"
  top: "patch_2_conv1-2_1"
}
layer {
  name: "patch_2_conv1-2_2"
  type: "Convolution"
  bottom: "patch_2_conv1-2_1"
  top: "patch_2_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv1-2_2_w"
 #param: "patch_2_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_2_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv1-2_2"
  top: "patch_2_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv1-2_2"
  type: "ReLU"
  bottom: "patch_2_conv1-2_2"
  top: "patch_2_conv1-2_2"
}
layer {
  name: "patch_2_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_2_conv1-2_2"
  bottom:"patch_2_shortcut1-1"
  top: "patch_2_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv1-3_1"
  type: "Convolution"
  bottom: "patch_2_shortcut1-2"
  top: "patch_2_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv1-3_1_w"
 #param: "patch_2_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_2_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv1-3_1"
  top: "patch_2_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv1-3_1"
  type: "ReLU"
  bottom: "patch_2_conv1-3_1"
  top: "patch_2_conv1-3_1"
}
layer {
  name: "patch_2_conv1-3_2"
  type: "Convolution"
  bottom: "patch_2_conv1-3_1"
  top: "patch_2_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv1-3_2_w"
 #param: "patch_2_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_2_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv1-3_2"
  top: "patch_2_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv1-3_2"
  type: "ReLU"
  bottom: "patch_2_conv1-3_2"
  top: "patch_2_conv1-3_2"
}
layer {
  name: "patch_2_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_2_conv1-3_2"
  bottom:"patch_2_shortcut1-2"
  top: "patch_2_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv2-1_1"
  type: "Convolution"
  bottom: "patch_2_shortcut1-3"
  top: "patch_2_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-1_1_w"
 #param: "patch_2_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_2_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-1_1"
  top: "patch_2_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-1_1"
  type: "ReLU"
  bottom: "patch_2_conv2-1_1"
  top: "patch_2_conv2-1_1"
}
layer {
  name: "patch_2_conv2-1_2"
  type: "Convolution"
  bottom: "patch_2_conv2-1_1"
  top: "patch_2_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-1_2_w"
 #param: "patch_2_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_2_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-1_2"
  top: "patch_2_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-1_2"
  type: "ReLU"
  bottom: "patch_2_conv2-1_2"
  top: "patch_2_conv2-1_2"
}
layer {
  name: "patch_2_project1"
  type: "Convolution"
  bottom: "patch_2_shortcut1-3"
  top: "patch_2_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_project1_w"
 #param: "patch_2_project1_b"
}
layer {
  name: "batch_nor_patch_2_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_project1"
  top: "patch_2_project1"
  
  
  
}
layer {
  name: "relu_patch_2_project1"
  type: "ReLU"
  bottom: "patch_2_project1"
  top: "patch_2_project1"
}
layer {
  name: "patch_2_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_2_conv2-1_2"
  bottom:"patch_2_project1"
  top: "patch_2_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv2-2_1"
  type: "Convolution"
  bottom: "patch_2_shortcut2-1"
  top: "patch_2_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-2_1_w"
 #param: "patch_2_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_2_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-2_1"
  top: "patch_2_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-2_1"
  type: "ReLU"
  bottom: "patch_2_conv2-2_1"
  top: "patch_2_conv2-2_1"
}
layer {
  name: "patch_2_conv2-2_2"
  type: "Convolution"
  bottom: "patch_2_conv2-2_1"
  top: "patch_2_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-2_2_w"
 #param: "patch_2_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_2_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-2_2"
  top: "patch_2_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-2_2"
  type: "ReLU"
  bottom: "patch_2_conv2-2_2"
  top: "patch_2_conv2-2_2"
}
layer {
  name: "patch_2_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_2_conv2-2_2"
  bottom:"patch_2_shortcut2-1"
  top: "patch_2_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv2-3_1"
  type: "Convolution"
  bottom: "patch_2_shortcut2-2"
  top: "patch_2_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-3_1_w"
 #param: "patch_2_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_2_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-3_1"
  top: "patch_2_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-3_1"
  type: "ReLU"
  bottom: "patch_2_conv2-3_1"
  top: "patch_2_conv2-3_1"
}
layer {
  name: "patch_2_conv2-3_2"
  type: "Convolution"
  bottom: "patch_2_conv2-3_1"
  top: "patch_2_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-3_2_w"
 #param: "patch_2_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_2_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-3_2"
  top: "patch_2_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-3_2"
  type: "ReLU"
  bottom: "patch_2_conv2-3_2"
  top: "patch_2_conv2-3_2"
}
layer {
  name: "patch_2_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_2_conv2-3_2"
  bottom:"patch_2_shortcut2-2"
  top: "patch_2_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv2-4_1"
  type: "Convolution"
  bottom: "patch_2_shortcut2-3"
  top: "patch_2_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-4_1_w"
 #param: "patch_2_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_2_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-4_1"
  top: "patch_2_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-4_1"
  type: "ReLU"
  bottom: "patch_2_conv2-4_1"
  top: "patch_2_conv2-4_1"
}
layer {
  name: "patch_2_conv2-4_2"
  type: "Convolution"
  bottom: "patch_2_conv2-4_1"
  top: "patch_2_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv2-4_2_w"
 #param: "patch_2_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_2_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv2-4_2"
  top: "patch_2_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv2-4_2"
  type: "ReLU"
  bottom: "patch_2_conv2-4_2"
  top: "patch_2_conv2-4_2"
}
layer {
  name: "patch_2_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_2_conv2-4_2"
  bottom:"patch_2_shortcut2-3"
  top: "patch_2_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv3-1_1"
  type: "Convolution"
  bottom: "patch_2_shortcut2-4"
  top: "patch_2_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-1_1_w"
 #param: "patch_2_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_2_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-1_1"
  top: "patch_2_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-1_1"
  type: "ReLU"
  bottom: "patch_2_conv3-1_1"
  top: "patch_2_conv3-1_1"
}
layer {
  name: "patch_2_conv3-1_2"
  type: "Convolution"
  bottom: "patch_2_conv3-1_1"
  top: "patch_2_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-1_2_w"
 #param: "patch_2_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_2_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-1_2"
  top: "patch_2_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-1_2"
  type: "ReLU"
  bottom: "patch_2_conv3-1_2"
  top: "patch_2_conv3-1_2"
}
layer {
  name: "patch_2_project2"
  type: "Convolution"
  bottom: "patch_2_shortcut2-4"
  top: "patch_2_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_project2_w"
 #param: "patch_2_project2_b"
}
layer {
  name: "batch_nor_patch_2_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_project2"
  top: "patch_2_project2"
  
  
  
}
layer {
  name: "relu_patch_2_project2"
  type: "ReLU"
  bottom: "patch_2_project2"
  top: "patch_2_project2"
}
layer {
  name: "patch_2_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_2_conv3-1_2"
  bottom:"patch_2_project2"
  top: "patch_2_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv3-2_1"
  type: "Convolution"
  bottom: "patch_2_shortcut3-1"
  top: "patch_2_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-2_1_w"
 #param: "patch_2_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_2_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-2_1"
  top: "patch_2_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-2_1"
  type: "ReLU"
  bottom: "patch_2_conv3-2_1"
  top: "patch_2_conv3-2_1"
}
layer {
  name: "patch_2_conv3-2_2"
  type: "Convolution"
  bottom: "patch_2_conv3-2_1"
  top: "patch_2_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-2_2_w"
 #param: "patch_2_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_2_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-2_2"
  top: "patch_2_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-2_2"
  type: "ReLU"
  bottom: "patch_2_conv3-2_2"
  top: "patch_2_conv3-2_2"
}
layer {
  name: "patch_2_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_2_conv3-2_2"
  bottom:"patch_2_shortcut3-1"
  top: "patch_2_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv3-3_1"
  type: "Convolution"
  bottom: "patch_2_shortcut3-2"
  top: "patch_2_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-3_1_w"
 #param: "patch_2_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_2_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-3_1"
  top: "patch_2_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-3_1"
  type: "ReLU"
  bottom: "patch_2_conv3-3_1"
  top: "patch_2_conv3-3_1"
}
layer {
  name: "patch_2_conv3-3_2"
  type: "Convolution"
  bottom: "patch_2_conv3-3_1"
  top: "patch_2_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-3_2_w"
 #param: "patch_2_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_2_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-3_2"
  top: "patch_2_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-3_2"
  type: "ReLU"
  bottom: "patch_2_conv3-3_2"
  top: "patch_2_conv3-3_2"
}
layer {
  name: "patch_2_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_2_conv3-3_2"
  bottom:"patch_2_shortcut3-2"
  top: "patch_2_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv3-4_1"
  type: "Convolution"
  bottom: "patch_2_shortcut3-3"
  top: "patch_2_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-4_1_w"
 #param: "patch_2_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_2_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-4_1"
  top: "patch_2_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-4_1"
  type: "ReLU"
  bottom: "patch_2_conv3-4_1"
  top: "patch_2_conv3-4_1"
}
layer {
  name: "patch_2_conv3-4_2"
  type: "Convolution"
  bottom: "patch_2_conv3-4_1"
  top: "patch_2_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-4_2_w"
 #param: "patch_2_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_2_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-4_2"
  top: "patch_2_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-4_2"
  type: "ReLU"
  bottom: "patch_2_conv3-4_2"
  top: "patch_2_conv3-4_2"
}
layer {
  name: "patch_2_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_2_conv3-4_2"
  bottom:"patch_2_shortcut3-3"
  top: "patch_2_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv3-5_1"
  type: "Convolution"
  bottom: "patch_2_shortcut3-4"
  top: "patch_2_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-5_1_w"
 #param: "patch_2_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_2_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-5_1"
  top: "patch_2_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-5_1"
  type: "ReLU"
  bottom: "patch_2_conv3-5_1"
  top: "patch_2_conv3-5_1"
}
layer {
  name: "patch_2_conv3-5_2"
  type: "Convolution"
  bottom: "patch_2_conv3-5_1"
  top: "patch_2_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-5_2_w"
 #param: "patch_2_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_2_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-5_2"
  top: "patch_2_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-5_2"
  type: "ReLU"
  bottom: "patch_2_conv3-5_2"
  top: "patch_2_conv3-5_2"
}
layer {
  name: "patch_2_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_2_conv3-5_2"
  bottom:"patch_2_shortcut3-4"
  top: "patch_2_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv3-6_1"
  type: "Convolution"
  bottom: "patch_2_shortcut3-5"
  top: "patch_2_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-6_1_w"
 #param: "patch_2_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_2_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-6_1"
  top: "patch_2_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-6_1"
  type: "ReLU"
  bottom: "patch_2_conv3-6_1"
  top: "patch_2_conv3-6_1"
}
layer {
  name: "patch_2_conv3-6_2"
  type: "Convolution"
  bottom: "patch_2_conv3-6_1"
  top: "patch_2_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv3-6_2_w"
 #param: "patch_2_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_2_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv3-6_2"
  top: "patch_2_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv3-6_2"
  type: "ReLU"
  bottom: "patch_2_conv3-6_2"
  top: "patch_2_conv3-6_2"
}
layer {
  name: "patch_2_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_2_conv3-6_2"
  bottom:"patch_2_shortcut3-5"
  top: "patch_2_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv4-1_1"
  type: "Convolution"
  bottom: "patch_2_shortcut3-6"
  top: "patch_2_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv4-1_1_w"
 #param: "patch_2_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_2_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv4-1_1"
  top: "patch_2_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv4-1_1"
  type: "ReLU"
  bottom: "patch_2_conv4-1_1"
  top: "patch_2_conv4-1_1"
}
layer {
  name: "patch_2_conv4-1_2"
  type: "Convolution"
  bottom: "patch_2_conv4-1_1"
  top: "patch_2_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv4-1_2_w"
 #param: "patch_2_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_2_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv4-1_2"
  top: "patch_2_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv4-1_2"
  type: "ReLU"
  bottom: "patch_2_conv4-1_2"
  top: "patch_2_conv4-1_2"
}
layer {
  name: "patch_2_project3"
  type: "Convolution"
  bottom: "patch_2_shortcut3-6"
  top: "patch_2_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_project3_w"
 #param: "patch_2_project3_b"
}
layer {
  name: "batch_nor_patch_2_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_project3"
  top: "patch_2_project3"
  
  
  
}
layer {
  name: "relu_patch_2_project3"
  type: "ReLU"
  bottom: "patch_2_project3"
  top: "patch_2_project3"
}
layer {
  name: "patch_2_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_2_conv4-1_2"
  bottom:"patch_2_project3"
  top: "patch_2_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv4-2_1"
  type: "Convolution"
  bottom: "patch_2_shortcut4-1"
  top: "patch_2_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv4-2_1_w"
 #param: "patch_2_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_2_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv4-2_1"
  top: "patch_2_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv4-2_1"
  type: "ReLU"
  bottom: "patch_2_conv4-2_1"
  top: "patch_2_conv4-2_1"
}
layer {
  name: "patch_2_conv4-2_2"
  type: "Convolution"
  bottom: "patch_2_conv4-2_1"
  top: "patch_2_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv4-2_2_w"
 #param: "patch_2_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_2_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv4-2_2"
  top: "patch_2_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv4-2_2"
  type: "ReLU"
  bottom: "patch_2_conv4-2_2"
  top: "patch_2_conv4-2_2"
}
layer {
  name: "patch_2_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_2_conv4-2_2"
  bottom:"patch_2_shortcut4-1"
  top: "patch_2_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_conv4-3_1"
  type: "Convolution"
  bottom: "patch_2_shortcut4-2"
  top: "patch_2_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv4-3_1_w"
 #param: "patch_2_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_2_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv4-3_1"
  top: "patch_2_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_2_conv4-3_1"
  type: "ReLU"
  bottom: "patch_2_conv4-3_1"
  top: "patch_2_conv4-3_1"
}
layer {
  name: "patch_2_conv4-3_2"
  type: "Convolution"
  bottom: "patch_2_conv4-3_1"
  top: "patch_2_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_conv4-3_2_w"
 #param: "patch_2_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_2_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_2_conv4-3_2"
  top: "patch_2_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_2_conv4-3_2"
  type: "ReLU"
  bottom: "patch_2_conv4-3_2"
  top: "patch_2_conv4-3_2"
}
layer {
  name: "patch_2_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_2_conv4-3_2"
  bottom:"patch_2_shortcut4-2"
  top: "patch_2_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_2_pool1"
  type: "Pooling"
  bottom: "patch_2_shortcut4-3"
  top: "patch_2_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_2_fc1"
  type: "InnerProduct"
  bottom: "patch_2_pool1"
  top: "patch_2_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_2_fc1_w"
 #param: "patch_2_fc1_b"
}

# start patch_3 network!
layer {
  name: "patch_3_conv0"
  type: "Convolution"
  bottom: "patch3"
  top: "patch_3_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv0_w"
 #param: "patch_3_conv0_b"
}
layer {
  name: "batch_nor_patch_3_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv0"
  top: "patch_3_conv0"
  
  
  
}
layer {
  name: "relu_patch_3_conv0"
  type: "ReLU"
  bottom: "patch_3_conv0"
  top: "patch_3_conv0"
}
layer {
  name: "patch_3_conv1-1_1"
  type: "Convolution"
  bottom: "patch_3_conv0"
  top: "patch_3_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv1-1_1_w"
 #param: "patch_3_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_3_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv1-1_1"
  top: "patch_3_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv1-1_1"
  type: "ReLU"
  bottom: "patch_3_conv1-1_1"
  top: "patch_3_conv1-1_1"
}
layer {
  name: "patch_3_conv1-1_2"
  type: "Convolution"
  bottom: "patch_3_conv1-1_1"
  top: "patch_3_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv1-1_2_w"
 #param: "patch_3_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_3_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv1-1_2"
  top: "patch_3_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv1-1_2"
  type: "ReLU"
  bottom: "patch_3_conv1-1_2"
  top: "patch_3_conv1-1_2"
}
layer {
  name: "patch_3_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_3_conv1-1_2"
  bottom:"patch_3_conv0"
  top: "patch_3_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv1-2_1"
  type: "Convolution"
  bottom: "patch_3_shortcut1-1"
  top: "patch_3_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv1-2_1_w"
 #param: "patch_3_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_3_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv1-2_1"
  top: "patch_3_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv1-2_1"
  type: "ReLU"
  bottom: "patch_3_conv1-2_1"
  top: "patch_3_conv1-2_1"
}
layer {
  name: "patch_3_conv1-2_2"
  type: "Convolution"
  bottom: "patch_3_conv1-2_1"
  top: "patch_3_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv1-2_2_w"
 #param: "patch_3_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_3_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv1-2_2"
  top: "patch_3_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv1-2_2"
  type: "ReLU"
  bottom: "patch_3_conv1-2_2"
  top: "patch_3_conv1-2_2"
}
layer {
  name: "patch_3_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_3_conv1-2_2"
  bottom:"patch_3_shortcut1-1"
  top: "patch_3_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv1-3_1"
  type: "Convolution"
  bottom: "patch_3_shortcut1-2"
  top: "patch_3_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv1-3_1_w"
 #param: "patch_3_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_3_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv1-3_1"
  top: "patch_3_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv1-3_1"
  type: "ReLU"
  bottom: "patch_3_conv1-3_1"
  top: "patch_3_conv1-3_1"
}
layer {
  name: "patch_3_conv1-3_2"
  type: "Convolution"
  bottom: "patch_3_conv1-3_1"
  top: "patch_3_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv1-3_2_w"
 #param: "patch_3_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_3_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv1-3_2"
  top: "patch_3_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv1-3_2"
  type: "ReLU"
  bottom: "patch_3_conv1-3_2"
  top: "patch_3_conv1-3_2"
}
layer {
  name: "patch_3_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_3_conv1-3_2"
  bottom:"patch_3_shortcut1-2"
  top: "patch_3_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv2-1_1"
  type: "Convolution"
  bottom: "patch_3_shortcut1-3"
  top: "patch_3_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-1_1_w"
 #param: "patch_3_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_3_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-1_1"
  top: "patch_3_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-1_1"
  type: "ReLU"
  bottom: "patch_3_conv2-1_1"
  top: "patch_3_conv2-1_1"
}
layer {
  name: "patch_3_conv2-1_2"
  type: "Convolution"
  bottom: "patch_3_conv2-1_1"
  top: "patch_3_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-1_2_w"
 #param: "patch_3_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_3_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-1_2"
  top: "patch_3_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-1_2"
  type: "ReLU"
  bottom: "patch_3_conv2-1_2"
  top: "patch_3_conv2-1_2"
}
layer {
  name: "patch_3_project1"
  type: "Convolution"
  bottom: "patch_3_shortcut1-3"
  top: "patch_3_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_project1_w"
 #param: "patch_3_project1_b"
}
layer {
  name: "batch_nor_patch_3_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_project1"
  top: "patch_3_project1"
  
  
  
}
layer {
  name: "relu_patch_3_project1"
  type: "ReLU"
  bottom: "patch_3_project1"
  top: "patch_3_project1"
}
layer {
  name: "patch_3_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_3_conv2-1_2"
  bottom:"patch_3_project1"
  top: "patch_3_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv2-2_1"
  type: "Convolution"
  bottom: "patch_3_shortcut2-1"
  top: "patch_3_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-2_1_w"
 #param: "patch_3_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_3_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-2_1"
  top: "patch_3_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-2_1"
  type: "ReLU"
  bottom: "patch_3_conv2-2_1"
  top: "patch_3_conv2-2_1"
}
layer {
  name: "patch_3_conv2-2_2"
  type: "Convolution"
  bottom: "patch_3_conv2-2_1"
  top: "patch_3_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-2_2_w"
 #param: "patch_3_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_3_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-2_2"
  top: "patch_3_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-2_2"
  type: "ReLU"
  bottom: "patch_3_conv2-2_2"
  top: "patch_3_conv2-2_2"
}
layer {
  name: "patch_3_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_3_conv2-2_2"
  bottom:"patch_3_shortcut2-1"
  top: "patch_3_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv2-3_1"
  type: "Convolution"
  bottom: "patch_3_shortcut2-2"
  top: "patch_3_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-3_1_w"
 #param: "patch_3_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_3_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-3_1"
  top: "patch_3_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-3_1"
  type: "ReLU"
  bottom: "patch_3_conv2-3_1"
  top: "patch_3_conv2-3_1"
}
layer {
  name: "patch_3_conv2-3_2"
  type: "Convolution"
  bottom: "patch_3_conv2-3_1"
  top: "patch_3_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-3_2_w"
 #param: "patch_3_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_3_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-3_2"
  top: "patch_3_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-3_2"
  type: "ReLU"
  bottom: "patch_3_conv2-3_2"
  top: "patch_3_conv2-3_2"
}
layer {
  name: "patch_3_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_3_conv2-3_2"
  bottom:"patch_3_shortcut2-2"
  top: "patch_3_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv2-4_1"
  type: "Convolution"
  bottom: "patch_3_shortcut2-3"
  top: "patch_3_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-4_1_w"
 #param: "patch_3_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_3_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-4_1"
  top: "patch_3_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-4_1"
  type: "ReLU"
  bottom: "patch_3_conv2-4_1"
  top: "patch_3_conv2-4_1"
}
layer {
  name: "patch_3_conv2-4_2"
  type: "Convolution"
  bottom: "patch_3_conv2-4_1"
  top: "patch_3_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv2-4_2_w"
 #param: "patch_3_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_3_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv2-4_2"
  top: "patch_3_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv2-4_2"
  type: "ReLU"
  bottom: "patch_3_conv2-4_2"
  top: "patch_3_conv2-4_2"
}
layer {
  name: "patch_3_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_3_conv2-4_2"
  bottom:"patch_3_shortcut2-3"
  top: "patch_3_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv3-1_1"
  type: "Convolution"
  bottom: "patch_3_shortcut2-4"
  top: "patch_3_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-1_1_w"
 #param: "patch_3_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_3_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-1_1"
  top: "patch_3_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-1_1"
  type: "ReLU"
  bottom: "patch_3_conv3-1_1"
  top: "patch_3_conv3-1_1"
}
layer {
  name: "patch_3_conv3-1_2"
  type: "Convolution"
  bottom: "patch_3_conv3-1_1"
  top: "patch_3_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-1_2_w"
 #param: "patch_3_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_3_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-1_2"
  top: "patch_3_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-1_2"
  type: "ReLU"
  bottom: "patch_3_conv3-1_2"
  top: "patch_3_conv3-1_2"
}
layer {
  name: "patch_3_project2"
  type: "Convolution"
  bottom: "patch_3_shortcut2-4"
  top: "patch_3_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_project2_w"
 #param: "patch_3_project2_b"
}
layer {
  name: "batch_nor_patch_3_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_project2"
  top: "patch_3_project2"
  
  
  
}
layer {
  name: "relu_patch_3_project2"
  type: "ReLU"
  bottom: "patch_3_project2"
  top: "patch_3_project2"
}
layer {
  name: "patch_3_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_3_conv3-1_2"
  bottom:"patch_3_project2"
  top: "patch_3_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv3-2_1"
  type: "Convolution"
  bottom: "patch_3_shortcut3-1"
  top: "patch_3_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-2_1_w"
 #param: "patch_3_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_3_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-2_1"
  top: "patch_3_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-2_1"
  type: "ReLU"
  bottom: "patch_3_conv3-2_1"
  top: "patch_3_conv3-2_1"
}
layer {
  name: "patch_3_conv3-2_2"
  type: "Convolution"
  bottom: "patch_3_conv3-2_1"
  top: "patch_3_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-2_2_w"
 #param: "patch_3_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_3_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-2_2"
  top: "patch_3_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-2_2"
  type: "ReLU"
  bottom: "patch_3_conv3-2_2"
  top: "patch_3_conv3-2_2"
}
layer {
  name: "patch_3_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_3_conv3-2_2"
  bottom:"patch_3_shortcut3-1"
  top: "patch_3_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv3-3_1"
  type: "Convolution"
  bottom: "patch_3_shortcut3-2"
  top: "patch_3_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-3_1_w"
 #param: "patch_3_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_3_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-3_1"
  top: "patch_3_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-3_1"
  type: "ReLU"
  bottom: "patch_3_conv3-3_1"
  top: "patch_3_conv3-3_1"
}
layer {
  name: "patch_3_conv3-3_2"
  type: "Convolution"
  bottom: "patch_3_conv3-3_1"
  top: "patch_3_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-3_2_w"
 #param: "patch_3_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_3_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-3_2"
  top: "patch_3_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-3_2"
  type: "ReLU"
  bottom: "patch_3_conv3-3_2"
  top: "patch_3_conv3-3_2"
}
layer {
  name: "patch_3_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_3_conv3-3_2"
  bottom:"patch_3_shortcut3-2"
  top: "patch_3_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv3-4_1"
  type: "Convolution"
  bottom: "patch_3_shortcut3-3"
  top: "patch_3_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-4_1_w"
 #param: "patch_3_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_3_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-4_1"
  top: "patch_3_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-4_1"
  type: "ReLU"
  bottom: "patch_3_conv3-4_1"
  top: "patch_3_conv3-4_1"
}
layer {
  name: "patch_3_conv3-4_2"
  type: "Convolution"
  bottom: "patch_3_conv3-4_1"
  top: "patch_3_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-4_2_w"
 #param: "patch_3_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_3_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-4_2"
  top: "patch_3_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-4_2"
  type: "ReLU"
  bottom: "patch_3_conv3-4_2"
  top: "patch_3_conv3-4_2"
}
layer {
  name: "patch_3_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_3_conv3-4_2"
  bottom:"patch_3_shortcut3-3"
  top: "patch_3_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv3-5_1"
  type: "Convolution"
  bottom: "patch_3_shortcut3-4"
  top: "patch_3_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-5_1_w"
 #param: "patch_3_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_3_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-5_1"
  top: "patch_3_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-5_1"
  type: "ReLU"
  bottom: "patch_3_conv3-5_1"
  top: "patch_3_conv3-5_1"
}
layer {
  name: "patch_3_conv3-5_2"
  type: "Convolution"
  bottom: "patch_3_conv3-5_1"
  top: "patch_3_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-5_2_w"
 #param: "patch_3_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_3_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-5_2"
  top: "patch_3_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-5_2"
  type: "ReLU"
  bottom: "patch_3_conv3-5_2"
  top: "patch_3_conv3-5_2"
}
layer {
  name: "patch_3_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_3_conv3-5_2"
  bottom:"patch_3_shortcut3-4"
  top: "patch_3_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv3-6_1"
  type: "Convolution"
  bottom: "patch_3_shortcut3-5"
  top: "patch_3_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-6_1_w"
 #param: "patch_3_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_3_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-6_1"
  top: "patch_3_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-6_1"
  type: "ReLU"
  bottom: "patch_3_conv3-6_1"
  top: "patch_3_conv3-6_1"
}
layer {
  name: "patch_3_conv3-6_2"
  type: "Convolution"
  bottom: "patch_3_conv3-6_1"
  top: "patch_3_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv3-6_2_w"
 #param: "patch_3_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_3_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv3-6_2"
  top: "patch_3_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv3-6_2"
  type: "ReLU"
  bottom: "patch_3_conv3-6_2"
  top: "patch_3_conv3-6_2"
}
layer {
  name: "patch_3_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_3_conv3-6_2"
  bottom:"patch_3_shortcut3-5"
  top: "patch_3_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv4-1_1"
  type: "Convolution"
  bottom: "patch_3_shortcut3-6"
  top: "patch_3_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv4-1_1_w"
 #param: "patch_3_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_3_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv4-1_1"
  top: "patch_3_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv4-1_1"
  type: "ReLU"
  bottom: "patch_3_conv4-1_1"
  top: "patch_3_conv4-1_1"
}
layer {
  name: "patch_3_conv4-1_2"
  type: "Convolution"
  bottom: "patch_3_conv4-1_1"
  top: "patch_3_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv4-1_2_w"
 #param: "patch_3_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_3_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv4-1_2"
  top: "patch_3_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv4-1_2"
  type: "ReLU"
  bottom: "patch_3_conv4-1_2"
  top: "patch_3_conv4-1_2"
}
layer {
  name: "patch_3_project3"
  type: "Convolution"
  bottom: "patch_3_shortcut3-6"
  top: "patch_3_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_project3_w"
 #param: "patch_3_project3_b"
}
layer {
  name: "batch_nor_patch_3_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_project3"
  top: "patch_3_project3"
  
  
  
}
layer {
  name: "relu_patch_3_project3"
  type: "ReLU"
  bottom: "patch_3_project3"
  top: "patch_3_project3"
}
layer {
  name: "patch_3_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_3_conv4-1_2"
  bottom:"patch_3_project3"
  top: "patch_3_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv4-2_1"
  type: "Convolution"
  bottom: "patch_3_shortcut4-1"
  top: "patch_3_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv4-2_1_w"
 #param: "patch_3_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_3_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv4-2_1"
  top: "patch_3_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv4-2_1"
  type: "ReLU"
  bottom: "patch_3_conv4-2_1"
  top: "patch_3_conv4-2_1"
}
layer {
  name: "patch_3_conv4-2_2"
  type: "Convolution"
  bottom: "patch_3_conv4-2_1"
  top: "patch_3_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv4-2_2_w"
 #param: "patch_3_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_3_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv4-2_2"
  top: "patch_3_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv4-2_2"
  type: "ReLU"
  bottom: "patch_3_conv4-2_2"
  top: "patch_3_conv4-2_2"
}
layer {
  name: "patch_3_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_3_conv4-2_2"
  bottom:"patch_3_shortcut4-1"
  top: "patch_3_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_conv4-3_1"
  type: "Convolution"
  bottom: "patch_3_shortcut4-2"
  top: "patch_3_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv4-3_1_w"
 #param: "patch_3_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_3_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv4-3_1"
  top: "patch_3_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_3_conv4-3_1"
  type: "ReLU"
  bottom: "patch_3_conv4-3_1"
  top: "patch_3_conv4-3_1"
}
layer {
  name: "patch_3_conv4-3_2"
  type: "Convolution"
  bottom: "patch_3_conv4-3_1"
  top: "patch_3_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_conv4-3_2_w"
 #param: "patch_3_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_3_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_3_conv4-3_2"
  top: "patch_3_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_3_conv4-3_2"
  type: "ReLU"
  bottom: "patch_3_conv4-3_2"
  top: "patch_3_conv4-3_2"
}
layer {
  name: "patch_3_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_3_conv4-3_2"
  bottom:"patch_3_shortcut4-2"
  top: "patch_3_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_3_pool1"
  type: "Pooling"
  bottom: "patch_3_shortcut4-3"
  top: "patch_3_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_3_fc1"
  type: "InnerProduct"
  bottom: "patch_3_pool1"
  top: "patch_3_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_3_fc1_w"
 #param: "patch_3_fc1_b"
}

# start patch_4 network!
layer {
  name: "patch_4_conv0"
  type: "Convolution"
  bottom: "patch4"
  top: "patch_4_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv0_w"
 #param: "patch_4_conv0_b"
}
layer {
  name: "batch_nor_patch_4_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv0"
  top: "patch_4_conv0"
  
  
  
}
layer {
  name: "relu_patch_4_conv0"
  type: "ReLU"
  bottom: "patch_4_conv0"
  top: "patch_4_conv0"
}
layer {
  name: "patch_4_conv1-1_1"
  type: "Convolution"
  bottom: "patch_4_conv0"
  top: "patch_4_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv1-1_1_w"
 #param: "patch_4_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_4_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv1-1_1"
  top: "patch_4_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv1-1_1"
  type: "ReLU"
  bottom: "patch_4_conv1-1_1"
  top: "patch_4_conv1-1_1"
}
layer {
  name: "patch_4_conv1-1_2"
  type: "Convolution"
  bottom: "patch_4_conv1-1_1"
  top: "patch_4_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv1-1_2_w"
 #param: "patch_4_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_4_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv1-1_2"
  top: "patch_4_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv1-1_2"
  type: "ReLU"
  bottom: "patch_4_conv1-1_2"
  top: "patch_4_conv1-1_2"
}
layer {
  name: "patch_4_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_4_conv1-1_2"
  bottom:"patch_4_conv0"
  top: "patch_4_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv1-2_1"
  type: "Convolution"
  bottom: "patch_4_shortcut1-1"
  top: "patch_4_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv1-2_1_w"
 #param: "patch_4_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_4_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv1-2_1"
  top: "patch_4_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv1-2_1"
  type: "ReLU"
  bottom: "patch_4_conv1-2_1"
  top: "patch_4_conv1-2_1"
}
layer {
  name: "patch_4_conv1-2_2"
  type: "Convolution"
  bottom: "patch_4_conv1-2_1"
  top: "patch_4_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv1-2_2_w"
 #param: "patch_4_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_4_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv1-2_2"
  top: "patch_4_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv1-2_2"
  type: "ReLU"
  bottom: "patch_4_conv1-2_2"
  top: "patch_4_conv1-2_2"
}
layer {
  name: "patch_4_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_4_conv1-2_2"
  bottom:"patch_4_shortcut1-1"
  top: "patch_4_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv1-3_1"
  type: "Convolution"
  bottom: "patch_4_shortcut1-2"
  top: "patch_4_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv1-3_1_w"
 #param: "patch_4_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_4_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv1-3_1"
  top: "patch_4_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv1-3_1"
  type: "ReLU"
  bottom: "patch_4_conv1-3_1"
  top: "patch_4_conv1-3_1"
}
layer {
  name: "patch_4_conv1-3_2"
  type: "Convolution"
  bottom: "patch_4_conv1-3_1"
  top: "patch_4_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv1-3_2_w"
 #param: "patch_4_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_4_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv1-3_2"
  top: "patch_4_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv1-3_2"
  type: "ReLU"
  bottom: "patch_4_conv1-3_2"
  top: "patch_4_conv1-3_2"
}
layer {
  name: "patch_4_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_4_conv1-3_2"
  bottom:"patch_4_shortcut1-2"
  top: "patch_4_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv2-1_1"
  type: "Convolution"
  bottom: "patch_4_shortcut1-3"
  top: "patch_4_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-1_1_w"
 #param: "patch_4_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_4_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-1_1"
  top: "patch_4_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-1_1"
  type: "ReLU"
  bottom: "patch_4_conv2-1_1"
  top: "patch_4_conv2-1_1"
}
layer {
  name: "patch_4_conv2-1_2"
  type: "Convolution"
  bottom: "patch_4_conv2-1_1"
  top: "patch_4_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-1_2_w"
 #param: "patch_4_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_4_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-1_2"
  top: "patch_4_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-1_2"
  type: "ReLU"
  bottom: "patch_4_conv2-1_2"
  top: "patch_4_conv2-1_2"
}
layer {
  name: "patch_4_project1"
  type: "Convolution"
  bottom: "patch_4_shortcut1-3"
  top: "patch_4_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_project1_w"
 #param: "patch_4_project1_b"
}
layer {
  name: "batch_nor_patch_4_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_project1"
  top: "patch_4_project1"
  
  
  
}
layer {
  name: "relu_patch_4_project1"
  type: "ReLU"
  bottom: "patch_4_project1"
  top: "patch_4_project1"
}
layer {
  name: "patch_4_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_4_conv2-1_2"
  bottom:"patch_4_project1"
  top: "patch_4_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv2-2_1"
  type: "Convolution"
  bottom: "patch_4_shortcut2-1"
  top: "patch_4_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-2_1_w"
 #param: "patch_4_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_4_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-2_1"
  top: "patch_4_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-2_1"
  type: "ReLU"
  bottom: "patch_4_conv2-2_1"
  top: "patch_4_conv2-2_1"
}
layer {
  name: "patch_4_conv2-2_2"
  type: "Convolution"
  bottom: "patch_4_conv2-2_1"
  top: "patch_4_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-2_2_w"
 #param: "patch_4_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_4_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-2_2"
  top: "patch_4_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-2_2"
  type: "ReLU"
  bottom: "patch_4_conv2-2_2"
  top: "patch_4_conv2-2_2"
}
layer {
  name: "patch_4_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_4_conv2-2_2"
  bottom:"patch_4_shortcut2-1"
  top: "patch_4_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv2-3_1"
  type: "Convolution"
  bottom: "patch_4_shortcut2-2"
  top: "patch_4_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-3_1_w"
 #param: "patch_4_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_4_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-3_1"
  top: "patch_4_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-3_1"
  type: "ReLU"
  bottom: "patch_4_conv2-3_1"
  top: "patch_4_conv2-3_1"
}
layer {
  name: "patch_4_conv2-3_2"
  type: "Convolution"
  bottom: "patch_4_conv2-3_1"
  top: "patch_4_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-3_2_w"
 #param: "patch_4_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_4_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-3_2"
  top: "patch_4_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-3_2"
  type: "ReLU"
  bottom: "patch_4_conv2-3_2"
  top: "patch_4_conv2-3_2"
}
layer {
  name: "patch_4_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_4_conv2-3_2"
  bottom:"patch_4_shortcut2-2"
  top: "patch_4_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv2-4_1"
  type: "Convolution"
  bottom: "patch_4_shortcut2-3"
  top: "patch_4_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-4_1_w"
 #param: "patch_4_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_4_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-4_1"
  top: "patch_4_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-4_1"
  type: "ReLU"
  bottom: "patch_4_conv2-4_1"
  top: "patch_4_conv2-4_1"
}
layer {
  name: "patch_4_conv2-4_2"
  type: "Convolution"
  bottom: "patch_4_conv2-4_1"
  top: "patch_4_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv2-4_2_w"
 #param: "patch_4_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_4_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv2-4_2"
  top: "patch_4_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv2-4_2"
  type: "ReLU"
  bottom: "patch_4_conv2-4_2"
  top: "patch_4_conv2-4_2"
}
layer {
  name: "patch_4_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_4_conv2-4_2"
  bottom:"patch_4_shortcut2-3"
  top: "patch_4_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv3-1_1"
  type: "Convolution"
  bottom: "patch_4_shortcut2-4"
  top: "patch_4_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-1_1_w"
 #param: "patch_4_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_4_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-1_1"
  top: "patch_4_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-1_1"
  type: "ReLU"
  bottom: "patch_4_conv3-1_1"
  top: "patch_4_conv3-1_1"
}
layer {
  name: "patch_4_conv3-1_2"
  type: "Convolution"
  bottom: "patch_4_conv3-1_1"
  top: "patch_4_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-1_2_w"
 #param: "patch_4_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_4_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-1_2"
  top: "patch_4_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-1_2"
  type: "ReLU"
  bottom: "patch_4_conv3-1_2"
  top: "patch_4_conv3-1_2"
}
layer {
  name: "patch_4_project2"
  type: "Convolution"
  bottom: "patch_4_shortcut2-4"
  top: "patch_4_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_project2_w"
 #param: "patch_4_project2_b"
}
layer {
  name: "batch_nor_patch_4_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_project2"
  top: "patch_4_project2"
  
  
  
}
layer {
  name: "relu_patch_4_project2"
  type: "ReLU"
  bottom: "patch_4_project2"
  top: "patch_4_project2"
}
layer {
  name: "patch_4_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_4_conv3-1_2"
  bottom:"patch_4_project2"
  top: "patch_4_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv3-2_1"
  type: "Convolution"
  bottom: "patch_4_shortcut3-1"
  top: "patch_4_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-2_1_w"
 #param: "patch_4_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_4_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-2_1"
  top: "patch_4_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-2_1"
  type: "ReLU"
  bottom: "patch_4_conv3-2_1"
  top: "patch_4_conv3-2_1"
}
layer {
  name: "patch_4_conv3-2_2"
  type: "Convolution"
  bottom: "patch_4_conv3-2_1"
  top: "patch_4_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-2_2_w"
 #param: "patch_4_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_4_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-2_2"
  top: "patch_4_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-2_2"
  type: "ReLU"
  bottom: "patch_4_conv3-2_2"
  top: "patch_4_conv3-2_2"
}
layer {
  name: "patch_4_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_4_conv3-2_2"
  bottom:"patch_4_shortcut3-1"
  top: "patch_4_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv3-3_1"
  type: "Convolution"
  bottom: "patch_4_shortcut3-2"
  top: "patch_4_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-3_1_w"
 #param: "patch_4_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_4_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-3_1"
  top: "patch_4_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-3_1"
  type: "ReLU"
  bottom: "patch_4_conv3-3_1"
  top: "patch_4_conv3-3_1"
}
layer {
  name: "patch_4_conv3-3_2"
  type: "Convolution"
  bottom: "patch_4_conv3-3_1"
  top: "patch_4_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-3_2_w"
 #param: "patch_4_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_4_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-3_2"
  top: "patch_4_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-3_2"
  type: "ReLU"
  bottom: "patch_4_conv3-3_2"
  top: "patch_4_conv3-3_2"
}
layer {
  name: "patch_4_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_4_conv3-3_2"
  bottom:"patch_4_shortcut3-2"
  top: "patch_4_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv3-4_1"
  type: "Convolution"
  bottom: "patch_4_shortcut3-3"
  top: "patch_4_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-4_1_w"
 #param: "patch_4_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_4_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-4_1"
  top: "patch_4_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-4_1"
  type: "ReLU"
  bottom: "patch_4_conv3-4_1"
  top: "patch_4_conv3-4_1"
}
layer {
  name: "patch_4_conv3-4_2"
  type: "Convolution"
  bottom: "patch_4_conv3-4_1"
  top: "patch_4_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-4_2_w"
 #param: "patch_4_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_4_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-4_2"
  top: "patch_4_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-4_2"
  type: "ReLU"
  bottom: "patch_4_conv3-4_2"
  top: "patch_4_conv3-4_2"
}
layer {
  name: "patch_4_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_4_conv3-4_2"
  bottom:"patch_4_shortcut3-3"
  top: "patch_4_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv3-5_1"
  type: "Convolution"
  bottom: "patch_4_shortcut3-4"
  top: "patch_4_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-5_1_w"
 #param: "patch_4_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_4_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-5_1"
  top: "patch_4_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-5_1"
  type: "ReLU"
  bottom: "patch_4_conv3-5_1"
  top: "patch_4_conv3-5_1"
}
layer {
  name: "patch_4_conv3-5_2"
  type: "Convolution"
  bottom: "patch_4_conv3-5_1"
  top: "patch_4_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-5_2_w"
 #param: "patch_4_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_4_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-5_2"
  top: "patch_4_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-5_2"
  type: "ReLU"
  bottom: "patch_4_conv3-5_2"
  top: "patch_4_conv3-5_2"
}
layer {
  name: "patch_4_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_4_conv3-5_2"
  bottom:"patch_4_shortcut3-4"
  top: "patch_4_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv3-6_1"
  type: "Convolution"
  bottom: "patch_4_shortcut3-5"
  top: "patch_4_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-6_1_w"
 #param: "patch_4_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_4_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-6_1"
  top: "patch_4_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-6_1"
  type: "ReLU"
  bottom: "patch_4_conv3-6_1"
  top: "patch_4_conv3-6_1"
}
layer {
  name: "patch_4_conv3-6_2"
  type: "Convolution"
  bottom: "patch_4_conv3-6_1"
  top: "patch_4_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv3-6_2_w"
 #param: "patch_4_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_4_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv3-6_2"
  top: "patch_4_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv3-6_2"
  type: "ReLU"
  bottom: "patch_4_conv3-6_2"
  top: "patch_4_conv3-6_2"
}
layer {
  name: "patch_4_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_4_conv3-6_2"
  bottom:"patch_4_shortcut3-5"
  top: "patch_4_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv4-1_1"
  type: "Convolution"
  bottom: "patch_4_shortcut3-6"
  top: "patch_4_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv4-1_1_w"
 #param: "patch_4_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_4_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv4-1_1"
  top: "patch_4_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv4-1_1"
  type: "ReLU"
  bottom: "patch_4_conv4-1_1"
  top: "patch_4_conv4-1_1"
}
layer {
  name: "patch_4_conv4-1_2"
  type: "Convolution"
  bottom: "patch_4_conv4-1_1"
  top: "patch_4_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv4-1_2_w"
 #param: "patch_4_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_4_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv4-1_2"
  top: "patch_4_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv4-1_2"
  type: "ReLU"
  bottom: "patch_4_conv4-1_2"
  top: "patch_4_conv4-1_2"
}
layer {
  name: "patch_4_project3"
  type: "Convolution"
  bottom: "patch_4_shortcut3-6"
  top: "patch_4_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_project3_w"
 #param: "patch_4_project3_b"
}
layer {
  name: "batch_nor_patch_4_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_project3"
  top: "patch_4_project3"
  
  
  
}
layer {
  name: "relu_patch_4_project3"
  type: "ReLU"
  bottom: "patch_4_project3"
  top: "patch_4_project3"
}
layer {
  name: "patch_4_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_4_conv4-1_2"
  bottom:"patch_4_project3"
  top: "patch_4_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv4-2_1"
  type: "Convolution"
  bottom: "patch_4_shortcut4-1"
  top: "patch_4_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv4-2_1_w"
 #param: "patch_4_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_4_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv4-2_1"
  top: "patch_4_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv4-2_1"
  type: "ReLU"
  bottom: "patch_4_conv4-2_1"
  top: "patch_4_conv4-2_1"
}
layer {
  name: "patch_4_conv4-2_2"
  type: "Convolution"
  bottom: "patch_4_conv4-2_1"
  top: "patch_4_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv4-2_2_w"
 #param: "patch_4_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_4_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv4-2_2"
  top: "patch_4_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv4-2_2"
  type: "ReLU"
  bottom: "patch_4_conv4-2_2"
  top: "patch_4_conv4-2_2"
}
layer {
  name: "patch_4_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_4_conv4-2_2"
  bottom:"patch_4_shortcut4-1"
  top: "patch_4_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_conv4-3_1"
  type: "Convolution"
  bottom: "patch_4_shortcut4-2"
  top: "patch_4_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv4-3_1_w"
 #param: "patch_4_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_4_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv4-3_1"
  top: "patch_4_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_4_conv4-3_1"
  type: "ReLU"
  bottom: "patch_4_conv4-3_1"
  top: "patch_4_conv4-3_1"
}
layer {
  name: "patch_4_conv4-3_2"
  type: "Convolution"
  bottom: "patch_4_conv4-3_1"
  top: "patch_4_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_conv4-3_2_w"
 #param: "patch_4_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_4_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_4_conv4-3_2"
  top: "patch_4_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_4_conv4-3_2"
  type: "ReLU"
  bottom: "patch_4_conv4-3_2"
  top: "patch_4_conv4-3_2"
}
layer {
  name: "patch_4_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_4_conv4-3_2"
  bottom:"patch_4_shortcut4-2"
  top: "patch_4_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_4_pool1"
  type: "Pooling"
  bottom: "patch_4_shortcut4-3"
  top: "patch_4_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_4_fc1"
  type: "InnerProduct"
  bottom: "patch_4_pool1"
  top: "patch_4_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_4_fc1_w"
 #param: "patch_4_fc1_b"
}

# start patch_5 network!
layer {
  name: "patch_5_conv0"
  type: "Convolution"
  bottom: "patch5"
  top: "patch_5_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv0_w"
 #param: "patch_5_conv0_b"
}
layer {
  name: "batch_nor_patch_5_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv0"
  top: "patch_5_conv0"
  
  
  
}
layer {
  name: "relu_patch_5_conv0"
  type: "ReLU"
  bottom: "patch_5_conv0"
  top: "patch_5_conv0"
}
layer {
  name: "patch_5_conv1-1_1"
  type: "Convolution"
  bottom: "patch_5_conv0"
  top: "patch_5_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv1-1_1_w"
 #param: "patch_5_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_5_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv1-1_1"
  top: "patch_5_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv1-1_1"
  type: "ReLU"
  bottom: "patch_5_conv1-1_1"
  top: "patch_5_conv1-1_1"
}
layer {
  name: "patch_5_conv1-1_2"
  type: "Convolution"
  bottom: "patch_5_conv1-1_1"
  top: "patch_5_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv1-1_2_w"
 #param: "patch_5_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_5_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv1-1_2"
  top: "patch_5_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv1-1_2"
  type: "ReLU"
  bottom: "patch_5_conv1-1_2"
  top: "patch_5_conv1-1_2"
}
layer {
  name: "patch_5_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_5_conv1-1_2"
  bottom:"patch_5_conv0"
  top: "patch_5_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv1-2_1"
  type: "Convolution"
  bottom: "patch_5_shortcut1-1"
  top: "patch_5_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv1-2_1_w"
 #param: "patch_5_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_5_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv1-2_1"
  top: "patch_5_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv1-2_1"
  type: "ReLU"
  bottom: "patch_5_conv1-2_1"
  top: "patch_5_conv1-2_1"
}
layer {
  name: "patch_5_conv1-2_2"
  type: "Convolution"
  bottom: "patch_5_conv1-2_1"
  top: "patch_5_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv1-2_2_w"
 #param: "patch_5_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_5_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv1-2_2"
  top: "patch_5_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv1-2_2"
  type: "ReLU"
  bottom: "patch_5_conv1-2_2"
  top: "patch_5_conv1-2_2"
}
layer {
  name: "patch_5_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_5_conv1-2_2"
  bottom:"patch_5_shortcut1-1"
  top: "patch_5_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv1-3_1"
  type: "Convolution"
  bottom: "patch_5_shortcut1-2"
  top: "patch_5_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv1-3_1_w"
 #param: "patch_5_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_5_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv1-3_1"
  top: "patch_5_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv1-3_1"
  type: "ReLU"
  bottom: "patch_5_conv1-3_1"
  top: "patch_5_conv1-3_1"
}
layer {
  name: "patch_5_conv1-3_2"
  type: "Convolution"
  bottom: "patch_5_conv1-3_1"
  top: "patch_5_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv1-3_2_w"
 #param: "patch_5_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_5_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv1-3_2"
  top: "patch_5_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv1-3_2"
  type: "ReLU"
  bottom: "patch_5_conv1-3_2"
  top: "patch_5_conv1-3_2"
}
layer {
  name: "patch_5_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_5_conv1-3_2"
  bottom:"patch_5_shortcut1-2"
  top: "patch_5_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv2-1_1"
  type: "Convolution"
  bottom: "patch_5_shortcut1-3"
  top: "patch_5_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-1_1_w"
 #param: "patch_5_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_5_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-1_1"
  top: "patch_5_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-1_1"
  type: "ReLU"
  bottom: "patch_5_conv2-1_1"
  top: "patch_5_conv2-1_1"
}
layer {
  name: "patch_5_conv2-1_2"
  type: "Convolution"
  bottom: "patch_5_conv2-1_1"
  top: "patch_5_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-1_2_w"
 #param: "patch_5_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_5_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-1_2"
  top: "patch_5_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-1_2"
  type: "ReLU"
  bottom: "patch_5_conv2-1_2"
  top: "patch_5_conv2-1_2"
}
layer {
  name: "patch_5_project1"
  type: "Convolution"
  bottom: "patch_5_shortcut1-3"
  top: "patch_5_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_project1_w"
 #param: "patch_5_project1_b"
}
layer {
  name: "batch_nor_patch_5_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_project1"
  top: "patch_5_project1"
  
  
  
}
layer {
  name: "relu_patch_5_project1"
  type: "ReLU"
  bottom: "patch_5_project1"
  top: "patch_5_project1"
}
layer {
  name: "patch_5_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_5_conv2-1_2"
  bottom:"patch_5_project1"
  top: "patch_5_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv2-2_1"
  type: "Convolution"
  bottom: "patch_5_shortcut2-1"
  top: "patch_5_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-2_1_w"
 #param: "patch_5_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_5_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-2_1"
  top: "patch_5_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-2_1"
  type: "ReLU"
  bottom: "patch_5_conv2-2_1"
  top: "patch_5_conv2-2_1"
}
layer {
  name: "patch_5_conv2-2_2"
  type: "Convolution"
  bottom: "patch_5_conv2-2_1"
  top: "patch_5_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-2_2_w"
 #param: "patch_5_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_5_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-2_2"
  top: "patch_5_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-2_2"
  type: "ReLU"
  bottom: "patch_5_conv2-2_2"
  top: "patch_5_conv2-2_2"
}
layer {
  name: "patch_5_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_5_conv2-2_2"
  bottom:"patch_5_shortcut2-1"
  top: "patch_5_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv2-3_1"
  type: "Convolution"
  bottom: "patch_5_shortcut2-2"
  top: "patch_5_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-3_1_w"
 #param: "patch_5_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_5_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-3_1"
  top: "patch_5_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-3_1"
  type: "ReLU"
  bottom: "patch_5_conv2-3_1"
  top: "patch_5_conv2-3_1"
}
layer {
  name: "patch_5_conv2-3_2"
  type: "Convolution"
  bottom: "patch_5_conv2-3_1"
  top: "patch_5_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-3_2_w"
 #param: "patch_5_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_5_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-3_2"
  top: "patch_5_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-3_2"
  type: "ReLU"
  bottom: "patch_5_conv2-3_2"
  top: "patch_5_conv2-3_2"
}
layer {
  name: "patch_5_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_5_conv2-3_2"
  bottom:"patch_5_shortcut2-2"
  top: "patch_5_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv2-4_1"
  type: "Convolution"
  bottom: "patch_5_shortcut2-3"
  top: "patch_5_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-4_1_w"
 #param: "patch_5_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_5_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-4_1"
  top: "patch_5_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-4_1"
  type: "ReLU"
  bottom: "patch_5_conv2-4_1"
  top: "patch_5_conv2-4_1"
}
layer {
  name: "patch_5_conv2-4_2"
  type: "Convolution"
  bottom: "patch_5_conv2-4_1"
  top: "patch_5_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv2-4_2_w"
 #param: "patch_5_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_5_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv2-4_2"
  top: "patch_5_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv2-4_2"
  type: "ReLU"
  bottom: "patch_5_conv2-4_2"
  top: "patch_5_conv2-4_2"
}
layer {
  name: "patch_5_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_5_conv2-4_2"
  bottom:"patch_5_shortcut2-3"
  top: "patch_5_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv3-1_1"
  type: "Convolution"
  bottom: "patch_5_shortcut2-4"
  top: "patch_5_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-1_1_w"
 #param: "patch_5_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_5_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-1_1"
  top: "patch_5_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-1_1"
  type: "ReLU"
  bottom: "patch_5_conv3-1_1"
  top: "patch_5_conv3-1_1"
}
layer {
  name: "patch_5_conv3-1_2"
  type: "Convolution"
  bottom: "patch_5_conv3-1_1"
  top: "patch_5_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-1_2_w"
 #param: "patch_5_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_5_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-1_2"
  top: "patch_5_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-1_2"
  type: "ReLU"
  bottom: "patch_5_conv3-1_2"
  top: "patch_5_conv3-1_2"
}
layer {
  name: "patch_5_project2"
  type: "Convolution"
  bottom: "patch_5_shortcut2-4"
  top: "patch_5_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_project2_w"
 #param: "patch_5_project2_b"
}
layer {
  name: "batch_nor_patch_5_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_project2"
  top: "patch_5_project2"
  
  
  
}
layer {
  name: "relu_patch_5_project2"
  type: "ReLU"
  bottom: "patch_5_project2"
  top: "patch_5_project2"
}
layer {
  name: "patch_5_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_5_conv3-1_2"
  bottom:"patch_5_project2"
  top: "patch_5_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv3-2_1"
  type: "Convolution"
  bottom: "patch_5_shortcut3-1"
  top: "patch_5_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-2_1_w"
 #param: "patch_5_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_5_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-2_1"
  top: "patch_5_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-2_1"
  type: "ReLU"
  bottom: "patch_5_conv3-2_1"
  top: "patch_5_conv3-2_1"
}
layer {
  name: "patch_5_conv3-2_2"
  type: "Convolution"
  bottom: "patch_5_conv3-2_1"
  top: "patch_5_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-2_2_w"
 #param: "patch_5_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_5_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-2_2"
  top: "patch_5_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-2_2"
  type: "ReLU"
  bottom: "patch_5_conv3-2_2"
  top: "patch_5_conv3-2_2"
}
layer {
  name: "patch_5_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_5_conv3-2_2"
  bottom:"patch_5_shortcut3-1"
  top: "patch_5_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv3-3_1"
  type: "Convolution"
  bottom: "patch_5_shortcut3-2"
  top: "patch_5_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-3_1_w"
 #param: "patch_5_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_5_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-3_1"
  top: "patch_5_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-3_1"
  type: "ReLU"
  bottom: "patch_5_conv3-3_1"
  top: "patch_5_conv3-3_1"
}
layer {
  name: "patch_5_conv3-3_2"
  type: "Convolution"
  bottom: "patch_5_conv3-3_1"
  top: "patch_5_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-3_2_w"
 #param: "patch_5_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_5_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-3_2"
  top: "patch_5_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-3_2"
  type: "ReLU"
  bottom: "patch_5_conv3-3_2"
  top: "patch_5_conv3-3_2"
}
layer {
  name: "patch_5_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_5_conv3-3_2"
  bottom:"patch_5_shortcut3-2"
  top: "patch_5_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv3-4_1"
  type: "Convolution"
  bottom: "patch_5_shortcut3-3"
  top: "patch_5_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-4_1_w"
 #param: "patch_5_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_5_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-4_1"
  top: "patch_5_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-4_1"
  type: "ReLU"
  bottom: "patch_5_conv3-4_1"
  top: "patch_5_conv3-4_1"
}
layer {
  name: "patch_5_conv3-4_2"
  type: "Convolution"
  bottom: "patch_5_conv3-4_1"
  top: "patch_5_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-4_2_w"
 #param: "patch_5_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_5_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-4_2"
  top: "patch_5_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-4_2"
  type: "ReLU"
  bottom: "patch_5_conv3-4_2"
  top: "patch_5_conv3-4_2"
}
layer {
  name: "patch_5_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_5_conv3-4_2"
  bottom:"patch_5_shortcut3-3"
  top: "patch_5_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv3-5_1"
  type: "Convolution"
  bottom: "patch_5_shortcut3-4"
  top: "patch_5_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-5_1_w"
 #param: "patch_5_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_5_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-5_1"
  top: "patch_5_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-5_1"
  type: "ReLU"
  bottom: "patch_5_conv3-5_1"
  top: "patch_5_conv3-5_1"
}
layer {
  name: "patch_5_conv3-5_2"
  type: "Convolution"
  bottom: "patch_5_conv3-5_1"
  top: "patch_5_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-5_2_w"
 #param: "patch_5_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_5_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-5_2"
  top: "patch_5_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-5_2"
  type: "ReLU"
  bottom: "patch_5_conv3-5_2"
  top: "patch_5_conv3-5_2"
}
layer {
  name: "patch_5_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_5_conv3-5_2"
  bottom:"patch_5_shortcut3-4"
  top: "patch_5_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv3-6_1"
  type: "Convolution"
  bottom: "patch_5_shortcut3-5"
  top: "patch_5_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-6_1_w"
 #param: "patch_5_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_5_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-6_1"
  top: "patch_5_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-6_1"
  type: "ReLU"
  bottom: "patch_5_conv3-6_1"
  top: "patch_5_conv3-6_1"
}
layer {
  name: "patch_5_conv3-6_2"
  type: "Convolution"
  bottom: "patch_5_conv3-6_1"
  top: "patch_5_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv3-6_2_w"
 #param: "patch_5_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_5_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv3-6_2"
  top: "patch_5_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv3-6_2"
  type: "ReLU"
  bottom: "patch_5_conv3-6_2"
  top: "patch_5_conv3-6_2"
}
layer {
  name: "patch_5_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_5_conv3-6_2"
  bottom:"patch_5_shortcut3-5"
  top: "patch_5_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv4-1_1"
  type: "Convolution"
  bottom: "patch_5_shortcut3-6"
  top: "patch_5_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv4-1_1_w"
 #param: "patch_5_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_5_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv4-1_1"
  top: "patch_5_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv4-1_1"
  type: "ReLU"
  bottom: "patch_5_conv4-1_1"
  top: "patch_5_conv4-1_1"
}
layer {
  name: "patch_5_conv4-1_2"
  type: "Convolution"
  bottom: "patch_5_conv4-1_1"
  top: "patch_5_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv4-1_2_w"
 #param: "patch_5_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_5_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv4-1_2"
  top: "patch_5_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv4-1_2"
  type: "ReLU"
  bottom: "patch_5_conv4-1_2"
  top: "patch_5_conv4-1_2"
}
layer {
  name: "patch_5_project3"
  type: "Convolution"
  bottom: "patch_5_shortcut3-6"
  top: "patch_5_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_project3_w"
 #param: "patch_5_project3_b"
}
layer {
  name: "batch_nor_patch_5_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_project3"
  top: "patch_5_project3"
  
  
  
}
layer {
  name: "relu_patch_5_project3"
  type: "ReLU"
  bottom: "patch_5_project3"
  top: "patch_5_project3"
}
layer {
  name: "patch_5_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_5_conv4-1_2"
  bottom:"patch_5_project3"
  top: "patch_5_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv4-2_1"
  type: "Convolution"
  bottom: "patch_5_shortcut4-1"
  top: "patch_5_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv4-2_1_w"
 #param: "patch_5_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_5_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv4-2_1"
  top: "patch_5_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv4-2_1"
  type: "ReLU"
  bottom: "patch_5_conv4-2_1"
  top: "patch_5_conv4-2_1"
}
layer {
  name: "patch_5_conv4-2_2"
  type: "Convolution"
  bottom: "patch_5_conv4-2_1"
  top: "patch_5_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv4-2_2_w"
 #param: "patch_5_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_5_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv4-2_2"
  top: "patch_5_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv4-2_2"
  type: "ReLU"
  bottom: "patch_5_conv4-2_2"
  top: "patch_5_conv4-2_2"
}
layer {
  name: "patch_5_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_5_conv4-2_2"
  bottom:"patch_5_shortcut4-1"
  top: "patch_5_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_conv4-3_1"
  type: "Convolution"
  bottom: "patch_5_shortcut4-2"
  top: "patch_5_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv4-3_1_w"
 #param: "patch_5_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_5_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv4-3_1"
  top: "patch_5_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_5_conv4-3_1"
  type: "ReLU"
  bottom: "patch_5_conv4-3_1"
  top: "patch_5_conv4-3_1"
}
layer {
  name: "patch_5_conv4-3_2"
  type: "Convolution"
  bottom: "patch_5_conv4-3_1"
  top: "patch_5_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_conv4-3_2_w"
 #param: "patch_5_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_5_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_5_conv4-3_2"
  top: "patch_5_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_5_conv4-3_2"
  type: "ReLU"
  bottom: "patch_5_conv4-3_2"
  top: "patch_5_conv4-3_2"
}
layer {
  name: "patch_5_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_5_conv4-3_2"
  bottom:"patch_5_shortcut4-2"
  top: "patch_5_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_5_pool1"
  type: "Pooling"
  bottom: "patch_5_shortcut4-3"
  top: "patch_5_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_5_fc1"
  type: "InnerProduct"
  bottom: "patch_5_pool1"
  top: "patch_5_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_5_fc1_w"
 #param: "patch_5_fc1_b"
}

# start patch_6 network!
layer {
  name: "patch_6_conv0"
  type: "Convolution"
  bottom: "patch6"
  top: "patch_6_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv0_w"
 #param: "patch_6_conv0_b"
}
layer {
  name: "batch_nor_patch_6_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv0"
  top: "patch_6_conv0"
  
  
  
}
layer {
  name: "relu_patch_6_conv0"
  type: "ReLU"
  bottom: "patch_6_conv0"
  top: "patch_6_conv0"
}
layer {
  name: "patch_6_conv1-1_1"
  type: "Convolution"
  bottom: "patch_6_conv0"
  top: "patch_6_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv1-1_1_w"
 #param: "patch_6_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_6_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv1-1_1"
  top: "patch_6_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv1-1_1"
  type: "ReLU"
  bottom: "patch_6_conv1-1_1"
  top: "patch_6_conv1-1_1"
}
layer {
  name: "patch_6_conv1-1_2"
  type: "Convolution"
  bottom: "patch_6_conv1-1_1"
  top: "patch_6_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv1-1_2_w"
 #param: "patch_6_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_6_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv1-1_2"
  top: "patch_6_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv1-1_2"
  type: "ReLU"
  bottom: "patch_6_conv1-1_2"
  top: "patch_6_conv1-1_2"
}
layer {
  name: "patch_6_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_6_conv1-1_2"
  bottom:"patch_6_conv0"
  top: "patch_6_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv1-2_1"
  type: "Convolution"
  bottom: "patch_6_shortcut1-1"
  top: "patch_6_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv1-2_1_w"
 #param: "patch_6_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_6_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv1-2_1"
  top: "patch_6_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv1-2_1"
  type: "ReLU"
  bottom: "patch_6_conv1-2_1"
  top: "patch_6_conv1-2_1"
}
layer {
  name: "patch_6_conv1-2_2"
  type: "Convolution"
  bottom: "patch_6_conv1-2_1"
  top: "patch_6_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv1-2_2_w"
 #param: "patch_6_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_6_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv1-2_2"
  top: "patch_6_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv1-2_2"
  type: "ReLU"
  bottom: "patch_6_conv1-2_2"
  top: "patch_6_conv1-2_2"
}
layer {
  name: "patch_6_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_6_conv1-2_2"
  bottom:"patch_6_shortcut1-1"
  top: "patch_6_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv1-3_1"
  type: "Convolution"
  bottom: "patch_6_shortcut1-2"
  top: "patch_6_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv1-3_1_w"
 #param: "patch_6_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_6_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv1-3_1"
  top: "patch_6_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv1-3_1"
  type: "ReLU"
  bottom: "patch_6_conv1-3_1"
  top: "patch_6_conv1-3_1"
}
layer {
  name: "patch_6_conv1-3_2"
  type: "Convolution"
  bottom: "patch_6_conv1-3_1"
  top: "patch_6_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv1-3_2_w"
 #param: "patch_6_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_6_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv1-3_2"
  top: "patch_6_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv1-3_2"
  type: "ReLU"
  bottom: "patch_6_conv1-3_2"
  top: "patch_6_conv1-3_2"
}
layer {
  name: "patch_6_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_6_conv1-3_2"
  bottom:"patch_6_shortcut1-2"
  top: "patch_6_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv2-1_1"
  type: "Convolution"
  bottom: "patch_6_shortcut1-3"
  top: "patch_6_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-1_1_w"
 #param: "patch_6_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_6_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-1_1"
  top: "patch_6_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-1_1"
  type: "ReLU"
  bottom: "patch_6_conv2-1_1"
  top: "patch_6_conv2-1_1"
}
layer {
  name: "patch_6_conv2-1_2"
  type: "Convolution"
  bottom: "patch_6_conv2-1_1"
  top: "patch_6_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-1_2_w"
 #param: "patch_6_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_6_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-1_2"
  top: "patch_6_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-1_2"
  type: "ReLU"
  bottom: "patch_6_conv2-1_2"
  top: "patch_6_conv2-1_2"
}
layer {
  name: "patch_6_project1"
  type: "Convolution"
  bottom: "patch_6_shortcut1-3"
  top: "patch_6_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_project1_w"
 #param: "patch_6_project1_b"
}
layer {
  name: "batch_nor_patch_6_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_project1"
  top: "patch_6_project1"
  
  
  
}
layer {
  name: "relu_patch_6_project1"
  type: "ReLU"
  bottom: "patch_6_project1"
  top: "patch_6_project1"
}
layer {
  name: "patch_6_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_6_conv2-1_2"
  bottom:"patch_6_project1"
  top: "patch_6_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv2-2_1"
  type: "Convolution"
  bottom: "patch_6_shortcut2-1"
  top: "patch_6_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-2_1_w"
 #param: "patch_6_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_6_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-2_1"
  top: "patch_6_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-2_1"
  type: "ReLU"
  bottom: "patch_6_conv2-2_1"
  top: "patch_6_conv2-2_1"
}
layer {
  name: "patch_6_conv2-2_2"
  type: "Convolution"
  bottom: "patch_6_conv2-2_1"
  top: "patch_6_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-2_2_w"
 #param: "patch_6_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_6_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-2_2"
  top: "patch_6_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-2_2"
  type: "ReLU"
  bottom: "patch_6_conv2-2_2"
  top: "patch_6_conv2-2_2"
}
layer {
  name: "patch_6_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_6_conv2-2_2"
  bottom:"patch_6_shortcut2-1"
  top: "patch_6_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv2-3_1"
  type: "Convolution"
  bottom: "patch_6_shortcut2-2"
  top: "patch_6_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-3_1_w"
 #param: "patch_6_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_6_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-3_1"
  top: "patch_6_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-3_1"
  type: "ReLU"
  bottom: "patch_6_conv2-3_1"
  top: "patch_6_conv2-3_1"
}
layer {
  name: "patch_6_conv2-3_2"
  type: "Convolution"
  bottom: "patch_6_conv2-3_1"
  top: "patch_6_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-3_2_w"
 #param: "patch_6_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_6_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-3_2"
  top: "patch_6_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-3_2"
  type: "ReLU"
  bottom: "patch_6_conv2-3_2"
  top: "patch_6_conv2-3_2"
}
layer {
  name: "patch_6_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_6_conv2-3_2"
  bottom:"patch_6_shortcut2-2"
  top: "patch_6_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv2-4_1"
  type: "Convolution"
  bottom: "patch_6_shortcut2-3"
  top: "patch_6_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-4_1_w"
 #param: "patch_6_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_6_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-4_1"
  top: "patch_6_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-4_1"
  type: "ReLU"
  bottom: "patch_6_conv2-4_1"
  top: "patch_6_conv2-4_1"
}
layer {
  name: "patch_6_conv2-4_2"
  type: "Convolution"
  bottom: "patch_6_conv2-4_1"
  top: "patch_6_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv2-4_2_w"
 #param: "patch_6_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_6_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv2-4_2"
  top: "patch_6_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv2-4_2"
  type: "ReLU"
  bottom: "patch_6_conv2-4_2"
  top: "patch_6_conv2-4_2"
}
layer {
  name: "patch_6_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_6_conv2-4_2"
  bottom:"patch_6_shortcut2-3"
  top: "patch_6_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv3-1_1"
  type: "Convolution"
  bottom: "patch_6_shortcut2-4"
  top: "patch_6_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-1_1_w"
 #param: "patch_6_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_6_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-1_1"
  top: "patch_6_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-1_1"
  type: "ReLU"
  bottom: "patch_6_conv3-1_1"
  top: "patch_6_conv3-1_1"
}
layer {
  name: "patch_6_conv3-1_2"
  type: "Convolution"
  bottom: "patch_6_conv3-1_1"
  top: "patch_6_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-1_2_w"
 #param: "patch_6_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_6_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-1_2"
  top: "patch_6_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-1_2"
  type: "ReLU"
  bottom: "patch_6_conv3-1_2"
  top: "patch_6_conv3-1_2"
}
layer {
  name: "patch_6_project2"
  type: "Convolution"
  bottom: "patch_6_shortcut2-4"
  top: "patch_6_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_project2_w"
 #param: "patch_6_project2_b"
}
layer {
  name: "batch_nor_patch_6_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_project2"
  top: "patch_6_project2"
  
  
  
}
layer {
  name: "relu_patch_6_project2"
  type: "ReLU"
  bottom: "patch_6_project2"
  top: "patch_6_project2"
}
layer {
  name: "patch_6_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_6_conv3-1_2"
  bottom:"patch_6_project2"
  top: "patch_6_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv3-2_1"
  type: "Convolution"
  bottom: "patch_6_shortcut3-1"
  top: "patch_6_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-2_1_w"
 #param: "patch_6_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_6_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-2_1"
  top: "patch_6_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-2_1"
  type: "ReLU"
  bottom: "patch_6_conv3-2_1"
  top: "patch_6_conv3-2_1"
}
layer {
  name: "patch_6_conv3-2_2"
  type: "Convolution"
  bottom: "patch_6_conv3-2_1"
  top: "patch_6_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-2_2_w"
 #param: "patch_6_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_6_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-2_2"
  top: "patch_6_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-2_2"
  type: "ReLU"
  bottom: "patch_6_conv3-2_2"
  top: "patch_6_conv3-2_2"
}
layer {
  name: "patch_6_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_6_conv3-2_2"
  bottom:"patch_6_shortcut3-1"
  top: "patch_6_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv3-3_1"
  type: "Convolution"
  bottom: "patch_6_shortcut3-2"
  top: "patch_6_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-3_1_w"
 #param: "patch_6_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_6_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-3_1"
  top: "patch_6_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-3_1"
  type: "ReLU"
  bottom: "patch_6_conv3-3_1"
  top: "patch_6_conv3-3_1"
}
layer {
  name: "patch_6_conv3-3_2"
  type: "Convolution"
  bottom: "patch_6_conv3-3_1"
  top: "patch_6_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-3_2_w"
 #param: "patch_6_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_6_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-3_2"
  top: "patch_6_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-3_2"
  type: "ReLU"
  bottom: "patch_6_conv3-3_2"
  top: "patch_6_conv3-3_2"
}
layer {
  name: "patch_6_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_6_conv3-3_2"
  bottom:"patch_6_shortcut3-2"
  top: "patch_6_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv3-4_1"
  type: "Convolution"
  bottom: "patch_6_shortcut3-3"
  top: "patch_6_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-4_1_w"
 #param: "patch_6_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_6_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-4_1"
  top: "patch_6_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-4_1"
  type: "ReLU"
  bottom: "patch_6_conv3-4_1"
  top: "patch_6_conv3-4_1"
}
layer {
  name: "patch_6_conv3-4_2"
  type: "Convolution"
  bottom: "patch_6_conv3-4_1"
  top: "patch_6_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-4_2_w"
 #param: "patch_6_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_6_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-4_2"
  top: "patch_6_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-4_2"
  type: "ReLU"
  bottom: "patch_6_conv3-4_2"
  top: "patch_6_conv3-4_2"
}
layer {
  name: "patch_6_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_6_conv3-4_2"
  bottom:"patch_6_shortcut3-3"
  top: "patch_6_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv3-5_1"
  type: "Convolution"
  bottom: "patch_6_shortcut3-4"
  top: "patch_6_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-5_1_w"
 #param: "patch_6_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_6_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-5_1"
  top: "patch_6_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-5_1"
  type: "ReLU"
  bottom: "patch_6_conv3-5_1"
  top: "patch_6_conv3-5_1"
}
layer {
  name: "patch_6_conv3-5_2"
  type: "Convolution"
  bottom: "patch_6_conv3-5_1"
  top: "patch_6_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-5_2_w"
 #param: "patch_6_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_6_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-5_2"
  top: "patch_6_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-5_2"
  type: "ReLU"
  bottom: "patch_6_conv3-5_2"
  top: "patch_6_conv3-5_2"
}
layer {
  name: "patch_6_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_6_conv3-5_2"
  bottom:"patch_6_shortcut3-4"
  top: "patch_6_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv3-6_1"
  type: "Convolution"
  bottom: "patch_6_shortcut3-5"
  top: "patch_6_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-6_1_w"
 #param: "patch_6_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_6_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-6_1"
  top: "patch_6_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-6_1"
  type: "ReLU"
  bottom: "patch_6_conv3-6_1"
  top: "patch_6_conv3-6_1"
}
layer {
  name: "patch_6_conv3-6_2"
  type: "Convolution"
  bottom: "patch_6_conv3-6_1"
  top: "patch_6_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv3-6_2_w"
 #param: "patch_6_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_6_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv3-6_2"
  top: "patch_6_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv3-6_2"
  type: "ReLU"
  bottom: "patch_6_conv3-6_2"
  top: "patch_6_conv3-6_2"
}
layer {
  name: "patch_6_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_6_conv3-6_2"
  bottom:"patch_6_shortcut3-5"
  top: "patch_6_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv4-1_1"
  type: "Convolution"
  bottom: "patch_6_shortcut3-6"
  top: "patch_6_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv4-1_1_w"
 #param: "patch_6_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_6_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv4-1_1"
  top: "patch_6_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv4-1_1"
  type: "ReLU"
  bottom: "patch_6_conv4-1_1"
  top: "patch_6_conv4-1_1"
}
layer {
  name: "patch_6_conv4-1_2"
  type: "Convolution"
  bottom: "patch_6_conv4-1_1"
  top: "patch_6_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv4-1_2_w"
 #param: "patch_6_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_6_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv4-1_2"
  top: "patch_6_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv4-1_2"
  type: "ReLU"
  bottom: "patch_6_conv4-1_2"
  top: "patch_6_conv4-1_2"
}
layer {
  name: "patch_6_project3"
  type: "Convolution"
  bottom: "patch_6_shortcut3-6"
  top: "patch_6_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_project3_w"
 #param: "patch_6_project3_b"
}
layer {
  name: "batch_nor_patch_6_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_project3"
  top: "patch_6_project3"
  
  
  
}
layer {
  name: "relu_patch_6_project3"
  type: "ReLU"
  bottom: "patch_6_project3"
  top: "patch_6_project3"
}
layer {
  name: "patch_6_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_6_conv4-1_2"
  bottom:"patch_6_project3"
  top: "patch_6_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv4-2_1"
  type: "Convolution"
  bottom: "patch_6_shortcut4-1"
  top: "patch_6_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv4-2_1_w"
 #param: "patch_6_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_6_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv4-2_1"
  top: "patch_6_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv4-2_1"
  type: "ReLU"
  bottom: "patch_6_conv4-2_1"
  top: "patch_6_conv4-2_1"
}
layer {
  name: "patch_6_conv4-2_2"
  type: "Convolution"
  bottom: "patch_6_conv4-2_1"
  top: "patch_6_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv4-2_2_w"
 #param: "patch_6_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_6_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv4-2_2"
  top: "patch_6_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv4-2_2"
  type: "ReLU"
  bottom: "patch_6_conv4-2_2"
  top: "patch_6_conv4-2_2"
}
layer {
  name: "patch_6_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_6_conv4-2_2"
  bottom:"patch_6_shortcut4-1"
  top: "patch_6_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_conv4-3_1"
  type: "Convolution"
  bottom: "patch_6_shortcut4-2"
  top: "patch_6_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv4-3_1_w"
 #param: "patch_6_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_6_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv4-3_1"
  top: "patch_6_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_6_conv4-3_1"
  type: "ReLU"
  bottom: "patch_6_conv4-3_1"
  top: "patch_6_conv4-3_1"
}
layer {
  name: "patch_6_conv4-3_2"
  type: "Convolution"
  bottom: "patch_6_conv4-3_1"
  top: "patch_6_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_conv4-3_2_w"
 #param: "patch_6_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_6_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_6_conv4-3_2"
  top: "patch_6_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_6_conv4-3_2"
  type: "ReLU"
  bottom: "patch_6_conv4-3_2"
  top: "patch_6_conv4-3_2"
}
layer {
  name: "patch_6_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_6_conv4-3_2"
  bottom:"patch_6_shortcut4-2"
  top: "patch_6_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_6_pool1"
  type: "Pooling"
  bottom: "patch_6_shortcut4-3"
  top: "patch_6_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_6_fc1"
  type: "InnerProduct"
  bottom: "patch_6_pool1"
  top: "patch_6_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_6_fc1_w"
 #param: "patch_6_fc1_b"
}

# start patch_0 network!
layer {
  name: "patch_0_conv0"
  type: "Convolution"
  bottom: "patch0"
  top: "patch_0_conv0"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv0_w"
 #param: "patch_0_conv0_b"
}
layer {
  name: "batch_nor_patch_0_conv0"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv0"
  top: "patch_0_conv0"
  
  
  
}
layer {
  name: "relu_patch_0_conv0"
  type: "ReLU"
  bottom: "patch_0_conv0"
  top: "patch_0_conv0"
}
layer {
  name: "patch_0_conv1-1_1"
  type: "Convolution"
  bottom: "patch_0_conv0"
  top: "patch_0_conv1-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv1-1_1_w"
 #param: "patch_0_conv1-1_1_b"
}
layer {
  name: "batch_nor_patch_0_conv1-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv1-1_1"
  top: "patch_0_conv1-1_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv1-1_1"
  type: "ReLU"
  bottom: "patch_0_conv1-1_1"
  top: "patch_0_conv1-1_1"
}
layer {
  name: "patch_0_conv1-1_2"
  type: "Convolution"
  bottom: "patch_0_conv1-1_1"
  top: "patch_0_conv1-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv1-1_2_w"
 #param: "patch_0_conv1-1_2_b"
}
layer {
  name: "batch_nor_patch_0_conv1-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv1-1_2"
  top: "patch_0_conv1-1_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv1-1_2"
  type: "ReLU"
  bottom: "patch_0_conv1-1_2"
  top: "patch_0_conv1-1_2"
}
layer {
  name: "patch_0_shortcut1-1"
  type: "Eltwise"
  bottom:"patch_0_conv1-1_2"
  bottom:"patch_0_conv0"
  top: "patch_0_shortcut1-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv1-2_1"
  type: "Convolution"
  bottom: "patch_0_shortcut1-1"
  top: "patch_0_conv1-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv1-2_1_w"
 #param: "patch_0_conv1-2_1_b"
}
layer {
  name: "batch_nor_patch_0_conv1-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv1-2_1"
  top: "patch_0_conv1-2_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv1-2_1"
  type: "ReLU"
  bottom: "patch_0_conv1-2_1"
  top: "patch_0_conv1-2_1"
}
layer {
  name: "patch_0_conv1-2_2"
  type: "Convolution"
  bottom: "patch_0_conv1-2_1"
  top: "patch_0_conv1-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv1-2_2_w"
 #param: "patch_0_conv1-2_2_b"
}
layer {
  name: "batch_nor_patch_0_conv1-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv1-2_2"
  top: "patch_0_conv1-2_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv1-2_2"
  type: "ReLU"
  bottom: "patch_0_conv1-2_2"
  top: "patch_0_conv1-2_2"
}
layer {
  name: "patch_0_shortcut1-2"
  type: "Eltwise"
  bottom:"patch_0_conv1-2_2"
  bottom:"patch_0_shortcut1-1"
  top: "patch_0_shortcut1-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv1-3_1"
  type: "Convolution"
  bottom: "patch_0_shortcut1-2"
  top: "patch_0_conv1-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv1-3_1_w"
 #param: "patch_0_conv1-3_1_b"
}
layer {
  name: "batch_nor_patch_0_conv1-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv1-3_1"
  top: "patch_0_conv1-3_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv1-3_1"
  type: "ReLU"
  bottom: "patch_0_conv1-3_1"
  top: "patch_0_conv1-3_1"
}
layer {
  name: "patch_0_conv1-3_2"
  type: "Convolution"
  bottom: "patch_0_conv1-3_1"
  top: "patch_0_conv1-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv1-3_2_w"
 #param: "patch_0_conv1-3_2_b"
}
layer {
  name: "batch_nor_patch_0_conv1-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv1-3_2"
  top: "patch_0_conv1-3_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv1-3_2"
  type: "ReLU"
  bottom: "patch_0_conv1-3_2"
  top: "patch_0_conv1-3_2"
}
layer {
  name: "patch_0_shortcut1-3"
  type: "Eltwise"
  bottom:"patch_0_conv1-3_2"
  bottom:"patch_0_shortcut1-2"
  top: "patch_0_shortcut1-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv2-1_1"
  type: "Convolution"
  bottom: "patch_0_shortcut1-3"
  top: "patch_0_conv2-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-1_1_w"
 #param: "patch_0_conv2-1_1_b"
}
layer {
  name: "batch_nor_patch_0_conv2-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-1_1"
  top: "patch_0_conv2-1_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-1_1"
  type: "ReLU"
  bottom: "patch_0_conv2-1_1"
  top: "patch_0_conv2-1_1"
}
layer {
  name: "patch_0_conv2-1_2"
  type: "Convolution"
  bottom: "patch_0_conv2-1_1"
  top: "patch_0_conv2-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-1_2_w"
 #param: "patch_0_conv2-1_2_b"
}
layer {
  name: "batch_nor_patch_0_conv2-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-1_2"
  top: "patch_0_conv2-1_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-1_2"
  type: "ReLU"
  bottom: "patch_0_conv2-1_2"
  top: "patch_0_conv2-1_2"
}
layer {
  name: "patch_0_project1"
  type: "Convolution"
  bottom: "patch_0_shortcut1-3"
  top: "patch_0_project1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_project1_w"
 #param: "patch_0_project1_b"
}
layer {
  name: "batch_nor_patch_0_project1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_project1"
  top: "patch_0_project1"
  
  
  
}
layer {
  name: "relu_patch_0_project1"
  type: "ReLU"
  bottom: "patch_0_project1"
  top: "patch_0_project1"
}
layer {
  name: "patch_0_shortcut2-1"
  type: "Eltwise"
  bottom:"patch_0_conv2-1_2"
  bottom:"patch_0_project1"
  top: "patch_0_shortcut2-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv2-2_1"
  type: "Convolution"
  bottom: "patch_0_shortcut2-1"
  top: "patch_0_conv2-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-2_1_w"
 #param: "patch_0_conv2-2_1_b"
}
layer {
  name: "batch_nor_patch_0_conv2-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-2_1"
  top: "patch_0_conv2-2_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-2_1"
  type: "ReLU"
  bottom: "patch_0_conv2-2_1"
  top: "patch_0_conv2-2_1"
}
layer {
  name: "patch_0_conv2-2_2"
  type: "Convolution"
  bottom: "patch_0_conv2-2_1"
  top: "patch_0_conv2-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-2_2_w"
 #param: "patch_0_conv2-2_2_b"
}
layer {
  name: "batch_nor_patch_0_conv2-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-2_2"
  top: "patch_0_conv2-2_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-2_2"
  type: "ReLU"
  bottom: "patch_0_conv2-2_2"
  top: "patch_0_conv2-2_2"
}
layer {
  name: "patch_0_shortcut2-2"
  type: "Eltwise"
  bottom:"patch_0_conv2-2_2"
  bottom:"patch_0_shortcut2-1"
  top: "patch_0_shortcut2-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv2-3_1"
  type: "Convolution"
  bottom: "patch_0_shortcut2-2"
  top: "patch_0_conv2-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-3_1_w"
 #param: "patch_0_conv2-3_1_b"
}
layer {
  name: "batch_nor_patch_0_conv2-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-3_1"
  top: "patch_0_conv2-3_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-3_1"
  type: "ReLU"
  bottom: "patch_0_conv2-3_1"
  top: "patch_0_conv2-3_1"
}
layer {
  name: "patch_0_conv2-3_2"
  type: "Convolution"
  bottom: "patch_0_conv2-3_1"
  top: "patch_0_conv2-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-3_2_w"
 #param: "patch_0_conv2-3_2_b"
}
layer {
  name: "batch_nor_patch_0_conv2-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-3_2"
  top: "patch_0_conv2-3_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-3_2"
  type: "ReLU"
  bottom: "patch_0_conv2-3_2"
  top: "patch_0_conv2-3_2"
}
layer {
  name: "patch_0_shortcut2-3"
  type: "Eltwise"
  bottom:"patch_0_conv2-3_2"
  bottom:"patch_0_shortcut2-2"
  top: "patch_0_shortcut2-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv2-4_1"
  type: "Convolution"
  bottom: "patch_0_shortcut2-3"
  top: "patch_0_conv2-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-4_1_w"
 #param: "patch_0_conv2-4_1_b"
}
layer {
  name: "batch_nor_patch_0_conv2-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-4_1"
  top: "patch_0_conv2-4_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-4_1"
  type: "ReLU"
  bottom: "patch_0_conv2-4_1"
  top: "patch_0_conv2-4_1"
}
layer {
  name: "patch_0_conv2-4_2"
  type: "Convolution"
  bottom: "patch_0_conv2-4_1"
  top: "patch_0_conv2-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 128
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv2-4_2_w"
 #param: "patch_0_conv2-4_2_b"
}
layer {
  name: "batch_nor_patch_0_conv2-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv2-4_2"
  top: "patch_0_conv2-4_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv2-4_2"
  type: "ReLU"
  bottom: "patch_0_conv2-4_2"
  top: "patch_0_conv2-4_2"
}
layer {
  name: "patch_0_shortcut2-4"
  type: "Eltwise"
  bottom:"patch_0_conv2-4_2"
  bottom:"patch_0_shortcut2-3"
  top: "patch_0_shortcut2-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv3-1_1"
  type: "Convolution"
  bottom: "patch_0_shortcut2-4"
  top: "patch_0_conv3-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-1_1_w"
 #param: "patch_0_conv3-1_1_b"
}
layer {
  name: "batch_nor_patch_0_conv3-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-1_1"
  top: "patch_0_conv3-1_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-1_1"
  type: "ReLU"
  bottom: "patch_0_conv3-1_1"
  top: "patch_0_conv3-1_1"
}
layer {
  name: "patch_0_conv3-1_2"
  type: "Convolution"
  bottom: "patch_0_conv3-1_1"
  top: "patch_0_conv3-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-1_2_w"
 #param: "patch_0_conv3-1_2_b"
}
layer {
  name: "batch_nor_patch_0_conv3-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-1_2"
  top: "patch_0_conv3-1_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-1_2"
  type: "ReLU"
  bottom: "patch_0_conv3-1_2"
  top: "patch_0_conv3-1_2"
}
layer {
  name: "patch_0_project2"
  type: "Convolution"
  bottom: "patch_0_shortcut2-4"
  top: "patch_0_project2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_project2_w"
 #param: "patch_0_project2_b"
}
layer {
  name: "batch_nor_patch_0_project2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_project2"
  top: "patch_0_project2"
  
  
  
}
layer {
  name: "relu_patch_0_project2"
  type: "ReLU"
  bottom: "patch_0_project2"
  top: "patch_0_project2"
}
layer {
  name: "patch_0_shortcut3-1"
  type: "Eltwise"
  bottom:"patch_0_conv3-1_2"
  bottom:"patch_0_project2"
  top: "patch_0_shortcut3-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv3-2_1"
  type: "Convolution"
  bottom: "patch_0_shortcut3-1"
  top: "patch_0_conv3-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-2_1_w"
 #param: "patch_0_conv3-2_1_b"
}
layer {
  name: "batch_nor_patch_0_conv3-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-2_1"
  top: "patch_0_conv3-2_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-2_1"
  type: "ReLU"
  bottom: "patch_0_conv3-2_1"
  top: "patch_0_conv3-2_1"
}
layer {
  name: "patch_0_conv3-2_2"
  type: "Convolution"
  bottom: "patch_0_conv3-2_1"
  top: "patch_0_conv3-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-2_2_w"
 #param: "patch_0_conv3-2_2_b"
}
layer {
  name: "batch_nor_patch_0_conv3-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-2_2"
  top: "patch_0_conv3-2_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-2_2"
  type: "ReLU"
  bottom: "patch_0_conv3-2_2"
  top: "patch_0_conv3-2_2"
}
layer {
  name: "patch_0_shortcut3-2"
  type: "Eltwise"
  bottom:"patch_0_conv3-2_2"
  bottom:"patch_0_shortcut3-1"
  top: "patch_0_shortcut3-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv3-3_1"
  type: "Convolution"
  bottom: "patch_0_shortcut3-2"
  top: "patch_0_conv3-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-3_1_w"
 #param: "patch_0_conv3-3_1_b"
}
layer {
  name: "batch_nor_patch_0_conv3-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-3_1"
  top: "patch_0_conv3-3_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-3_1"
  type: "ReLU"
  bottom: "patch_0_conv3-3_1"
  top: "patch_0_conv3-3_1"
}
layer {
  name: "patch_0_conv3-3_2"
  type: "Convolution"
  bottom: "patch_0_conv3-3_1"
  top: "patch_0_conv3-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-3_2_w"
 #param: "patch_0_conv3-3_2_b"
}
layer {
  name: "batch_nor_patch_0_conv3-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-3_2"
  top: "patch_0_conv3-3_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-3_2"
  type: "ReLU"
  bottom: "patch_0_conv3-3_2"
  top: "patch_0_conv3-3_2"
}
layer {
  name: "patch_0_shortcut3-3"
  type: "Eltwise"
  bottom:"patch_0_conv3-3_2"
  bottom:"patch_0_shortcut3-2"
  top: "patch_0_shortcut3-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv3-4_1"
  type: "Convolution"
  bottom: "patch_0_shortcut3-3"
  top: "patch_0_conv3-4_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-4_1_w"
 #param: "patch_0_conv3-4_1_b"
}
layer {
  name: "batch_nor_patch_0_conv3-4_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-4_1"
  top: "patch_0_conv3-4_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-4_1"
  type: "ReLU"
  bottom: "patch_0_conv3-4_1"
  top: "patch_0_conv3-4_1"
}
layer {
  name: "patch_0_conv3-4_2"
  type: "Convolution"
  bottom: "patch_0_conv3-4_1"
  top: "patch_0_conv3-4_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-4_2_w"
 #param: "patch_0_conv3-4_2_b"
}
layer {
  name: "batch_nor_patch_0_conv3-4_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-4_2"
  top: "patch_0_conv3-4_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-4_2"
  type: "ReLU"
  bottom: "patch_0_conv3-4_2"
  top: "patch_0_conv3-4_2"
}
layer {
  name: "patch_0_shortcut3-4"
  type: "Eltwise"
  bottom:"patch_0_conv3-4_2"
  bottom:"patch_0_shortcut3-3"
  top: "patch_0_shortcut3-4"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv3-5_1"
  type: "Convolution"
  bottom: "patch_0_shortcut3-4"
  top: "patch_0_conv3-5_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-5_1_w"
 #param: "patch_0_conv3-5_1_b"
}
layer {
  name: "batch_nor_patch_0_conv3-5_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-5_1"
  top: "patch_0_conv3-5_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-5_1"
  type: "ReLU"
  bottom: "patch_0_conv3-5_1"
  top: "patch_0_conv3-5_1"
}
layer {
  name: "patch_0_conv3-5_2"
  type: "Convolution"
  bottom: "patch_0_conv3-5_1"
  top: "patch_0_conv3-5_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-5_2_w"
 #param: "patch_0_conv3-5_2_b"
}
layer {
  name: "batch_nor_patch_0_conv3-5_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-5_2"
  top: "patch_0_conv3-5_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-5_2"
  type: "ReLU"
  bottom: "patch_0_conv3-5_2"
  top: "patch_0_conv3-5_2"
}
layer {
  name: "patch_0_shortcut3-5"
  type: "Eltwise"
  bottom:"patch_0_conv3-5_2"
  bottom:"patch_0_shortcut3-4"
  top: "patch_0_shortcut3-5"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv3-6_1"
  type: "Convolution"
  bottom: "patch_0_shortcut3-5"
  top: "patch_0_conv3-6_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-6_1_w"
 #param: "patch_0_conv3-6_1_b"
}
layer {
  name: "batch_nor_patch_0_conv3-6_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-6_1"
  top: "patch_0_conv3-6_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-6_1"
  type: "ReLU"
  bottom: "patch_0_conv3-6_1"
  top: "patch_0_conv3-6_1"
}
layer {
  name: "patch_0_conv3-6_2"
  type: "Convolution"
  bottom: "patch_0_conv3-6_1"
  top: "patch_0_conv3-6_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv3-6_2_w"
 #param: "patch_0_conv3-6_2_b"
}
layer {
  name: "batch_nor_patch_0_conv3-6_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv3-6_2"
  top: "patch_0_conv3-6_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv3-6_2"
  type: "ReLU"
  bottom: "patch_0_conv3-6_2"
  top: "patch_0_conv3-6_2"
}
layer {
  name: "patch_0_shortcut3-6"
  type: "Eltwise"
  bottom:"patch_0_conv3-6_2"
  bottom:"patch_0_shortcut3-5"
  top: "patch_0_shortcut3-6"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv4-1_1"
  type: "Convolution"
  bottom: "patch_0_shortcut3-6"
  top: "patch_0_conv4-1_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 2
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv4-1_1_w"
 #param: "patch_0_conv4-1_1_b"
}
layer {
  name: "batch_nor_patch_0_conv4-1_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv4-1_1"
  top: "patch_0_conv4-1_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv4-1_1"
  type: "ReLU"
  bottom: "patch_0_conv4-1_1"
  top: "patch_0_conv4-1_1"
}
layer {
  name: "patch_0_conv4-1_2"
  type: "Convolution"
  bottom: "patch_0_conv4-1_1"
  top: "patch_0_conv4-1_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv4-1_2_w"
 #param: "patch_0_conv4-1_2_b"
}
layer {
  name: "batch_nor_patch_0_conv4-1_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv4-1_2"
  top: "patch_0_conv4-1_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv4-1_2"
  type: "ReLU"
  bottom: "patch_0_conv4-1_2"
  top: "patch_0_conv4-1_2"
}
layer {
  name: "patch_0_project3"
  type: "Convolution"
  bottom: "patch_0_shortcut3-6"
  top: "patch_0_project3"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_project3_w"
 #param: "patch_0_project3_b"
}
layer {
  name: "batch_nor_patch_0_project3"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_project3"
  top: "patch_0_project3"
  
  
  
}
layer {
  name: "relu_patch_0_project3"
  type: "ReLU"
  bottom: "patch_0_project3"
  top: "patch_0_project3"
}
layer {
  name: "patch_0_shortcut4-1"
  type: "Eltwise"
  bottom:"patch_0_conv4-1_2"
  bottom:"patch_0_project3"
  top: "patch_0_shortcut4-1"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv4-2_1"
  type: "Convolution"
  bottom: "patch_0_shortcut4-1"
  top: "patch_0_conv4-2_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv4-2_1_w"
 #param: "patch_0_conv4-2_1_b"
}
layer {
  name: "batch_nor_patch_0_conv4-2_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv4-2_1"
  top: "patch_0_conv4-2_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv4-2_1"
  type: "ReLU"
  bottom: "patch_0_conv4-2_1"
  top: "patch_0_conv4-2_1"
}
layer {
  name: "patch_0_conv4-2_2"
  type: "Convolution"
  bottom: "patch_0_conv4-2_1"
  top: "patch_0_conv4-2_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv4-2_2_w"
 #param: "patch_0_conv4-2_2_b"
}
layer {
  name: "batch_nor_patch_0_conv4-2_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv4-2_2"
  top: "patch_0_conv4-2_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv4-2_2"
  type: "ReLU"
  bottom: "patch_0_conv4-2_2"
  top: "patch_0_conv4-2_2"
}
layer {
  name: "patch_0_shortcut4-2"
  type: "Eltwise"
  bottom:"patch_0_conv4-2_2"
  bottom:"patch_0_shortcut4-1"
  top: "patch_0_shortcut4-2"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_conv4-3_1"
  type: "Convolution"
  bottom: "patch_0_shortcut4-2"
  top: "patch_0_conv4-3_1"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv4-3_1_w"
 #param: "patch_0_conv4-3_1_b"
}
layer {
  name: "batch_nor_patch_0_conv4-3_1"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv4-3_1"
  top: "patch_0_conv4-3_1"
  
  
  
}
layer {
  name: "relu_patch_0_conv4-3_1"
  type: "ReLU"
  bottom: "patch_0_conv4-3_1"
  top: "patch_0_conv4-3_1"
}
layer {
  name: "patch_0_conv4-3_2"
  type: "Convolution"
  bottom: "patch_0_conv4-3_1"
  top: "patch_0_conv4-3_2"
  #blobs_lr: 1
  #blobs_lr: 2
  convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_conv4-3_2_w"
 #param: "patch_0_conv4-3_2_b"
}
layer {
  name: "batch_nor_patch_0_conv4-3_2"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: true 
  }
  bottom: "patch_0_conv4-3_2"
  top: "patch_0_conv4-3_2"
  
  
  
}
layer {
  name: "relu_patch_0_conv4-3_2"
  type: "ReLU"
  bottom: "patch_0_conv4-3_2"
  top: "patch_0_conv4-3_2"
}
layer {
  name: "patch_0_shortcut4-3"
  type: "Eltwise"
  bottom:"patch_0_conv4-3_2"
  bottom:"patch_0_shortcut4-2"
  top: "patch_0_shortcut4-3"
  eltwise_param {
    operation: SUM
    coeff: 1.0
    coeff: 1.0
  }
}
layer {
  name: "patch_0_pool1"
  type: "Pooling"
  bottom: "patch_0_shortcut4-3"
  top: "patch_0_pool1"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "patch_0_fc1"
  type: "InnerProduct"
  bottom: "patch_0_pool1"
  top: "patch_0_fc1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
    }
  }
 #param: "patch_0_fc1_w"
 #param: "patch_0_fc1_b"
}

layer {
  name: "concat"
  type: "Concat"
  bottom: "patch_1_fc1"
  bottom: "patch_2_fc1"
  bottom: "patch_3_fc1"
  bottom: "patch_4_fc1"
  bottom: "patch_5_fc1"
  bottom: "patch_6_fc1"
  bottom: "patch_0_fc1"
  top: "concat"
  concat_param {
    concat_dim: 1
  }
}

layer {
  name: "concat_batch_norm"
  type: "BatchNormKyle"
  batch_norm_param { use_alpha_beta: false 
  }
  bottom: "concat"
  top: "concat"
  
  
  
}


layer {
  name: "hc_1"
  type: "InnerProduct"
  bottom: "concat"
  top: "hc_1"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 4096 
    weight_filler {
      type: "gaussian"
      std: 0.003
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu_hc_1"
  type: "ReLU"
  bottom: "hc_1"
  top: "hc_1"
}
layer {
  name: "hc_2"
  type:  "InnerProduct"
  bottom: "hc_1"
  top: "hc_2"
  #blobs_lr: 1
  #blobs_lr: 2
  inner_product_param {
    num_output: 128 
    weight_filler {
      type: "gaussian"
      std: 0.003
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}

