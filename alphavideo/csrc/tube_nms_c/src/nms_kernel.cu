// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

int const bbox_dim = 15;


struct res
{
     float* bbox1_aligned;
     float* bbox2_aligned;
};

__device__ inline res align_bbox_on_frame(float const * const mid1, float const *const  bbox1, float const* t1, \

                                             float const * const mid2, float const * const bbox2, float const* t2, float const* mid_t){
  float d1 = abs(*t1 - *mid_t), d2 = abs(*t2 - *mid_t);
  float t = min(d1, d2);
  float bbox1_aligned[4], bbox2_aligned[4];
  if (d1 != 0){
    for (int i=0; i< 4; i++){
      bbox1_aligned[i] = mid1[i] * ((d1-t)/ d1) + bbox1[i] * (t/ d1);
    }
  }else{
    for (int i=0; i< 4; i++){
      bbox1_aligned[i] = mid1[i];
    }
  }
  if (d2 != 0){
    for (int i=0; i< 4; i++){
      bbox2_aligned[i] = mid2[i] * ((d2-t)/ d2) + bbox2[i] * (t/ d2);
    }
  }else{
    for (int i=0; i< 4; i++){
      bbox2_aligned[i] = mid2[i];
    }
  }
  res aligned_bbox = {bbox1_aligned, bbox2_aligned};
  return aligned_bbox;
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh, const float side_nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * (bbox_dim + 1)];
  if (threadIdx.x < col_size) {
    int d = 0;
    for (;d <= bbox_dim; d ++){
      block_boxes[threadIdx.x * (bbox_dim + 1) + d] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * (bbox_dim + 1) + d];
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * (bbox_dim + 1);
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (bbox_dim == 4){
          if (devIoU(cur_box, block_boxes + i * (bbox_dim + 1)) > nms_overlap_thresh) {
            t |= 1ULL << i;
          }
      }
      else if(bbox_dim == 15){
          const float *cur_box_mid = cur_box + 1;
          const float *cur_box_fr = cur_box + 6;
          const float *cur_box_bk = cur_box + 11;

          const float *block_boxes_mid = block_boxes + i * (bbox_dim + 1) + 1;
          const float *block_boxes_fr = block_boxes + i * (bbox_dim + 1) + 6;
          const float *block_boxes_bk = block_boxes + i * (bbox_dim + 1) + 11;

          res aligned_bbox_fr = align_bbox_on_frame(cur_box_mid, cur_box_fr, cur_box + 5,
                                           block_boxes_mid, block_boxes_fr, block_boxes + i * (bbox_dim + 1) + 5, cur_box);
          const float * cur_box_fr_aligned = aligned_bbox_fr.bbox1_aligned;
          const float * block_boxes_fr_aligned = aligned_bbox_fr.bbox2_aligned;

          res aligned_bbox_bk = align_bbox_on_frame(cur_box_mid, cur_box_bk, cur_box + 10,
                                           block_boxes_mid, block_boxes_bk, block_boxes + i * (bbox_dim + 1) + 10, cur_box);
          const float * cur_box_bk_aligned = aligned_bbox_bk.bbox1_aligned;
          const float * block_boxes_bk_aligned = aligned_bbox_bk.bbox2_aligned;

          if (devIoU(cur_box_mid, block_boxes_mid) > nms_overlap_thresh &&
              devIoU(cur_box_fr_aligned, block_boxes_fr_aligned) > side_nms_overlap_thresh &&
              devIoU(cur_box_bk_aligned, block_boxes_bk_aligned) > side_nms_overlap_thresh) {
            t |= 1ULL << i;
          }
      }


    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh, float side_nms_overlap_thresh) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, bbox_dim);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  side_nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}