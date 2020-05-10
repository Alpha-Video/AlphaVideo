#pragma once
#include <torch/extension.h>

at::Tensor ROIAlign3d_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor ROIAlign3d_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int length,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);

// Interface for Python
at::Tensor ROIAlign3d_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio) {
  if (input.type().is_cuda()) {
    return ROIAlign3d_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor ROIAlign3d_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int length,
                             const int height,
                             const int width,
                             const int sampling_ratio) {
  if (grad.type().is_cuda()) {
    return ROIAlign3d_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, length, height, width, sampling_ratio);
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_3d_forward",&ROIAlign3d_forward, "ROIAlign3d_forward");
  m.def("roi_align_3d_backward",&ROIAlign3d_backward, "ROIAlign3d_backward");
}