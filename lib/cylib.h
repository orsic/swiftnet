#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_map>


void add_confusion_matrix(uint32_t* y, uint32_t* yt, uint32_t size, uint64_t* matrix, uint32_t num_classes) {
  uint32_t target, i;
  for (i = 0; i < size; i++) {
    target = yt[i];
    if (target >= 0 && target < num_classes) {
      matrix[y[i]*num_classes + target] += 1;
    }
  }
}

void add_disp_class_error_number_matrix(uint32_t *y, uint32_t *yt, uint32_t *dt, uint32_t size, uint64_t* error_matrix,
uint64_t* num_matrix, uint32_t max_disp, uint32_t num_classes){
    uint32_t disp, label, pred, index, i;
    for(i =0; i < size; i++){
        disp = dt[i]; label = yt[i]; pred = y[i];
        if ((disp >= max_disp) | (label >= num_classes) | (pred >= num_classes)){continue;}
        index = label * max_disp + disp;
        if (label != pred)
            {error_matrix[index] += 1;}
        num_matrix[index] += 1;
    }
}


void impl_convert_colors_to_ids(int num_classes, int* color_data, int width, int height,
                                uint8_t* rgb_labels, uint8_t* id_labels, uint64_t* class_hist,
                                float max_wgt, float* class_weights, float* weights) {
  std::unordered_map<std::string, uint8_t> color_map;
  for (std::size_t i = 0; i < num_classes; i++) {
    int s = i * 4;
    std::ostringstream skey;
    for (int i = 0; i < 3; i++)
      skey << std::setw(3) << std::setfill('0') << color_data[s+i];

    //skey << std::setw(3) << std::setfill('0') << r;
    //std::cout << skey.str() << '\n';
    auto key = skey.str();
    color_map[key] = color_data[s+3];

  }
  //#pragma omp parallel for
  for (int r = 0; r < height; r++) {
    int stride = r * width * 3;
    for (int c = 0; c < width; c++) {
      std::ostringstream skey;
      for (int i = 0; i < 3; i++)
        skey << std::setw(3) << std::setfill('0') << int(rgb_labels[stride + c*3 + i]);
      auto key = skey.str();
      //std::cout << key << " - " << int(color_map[key]) << '\n';
      uint8_t class_id = color_map[key];
      id_labels[r*width + c] = class_id;
      if (class_id < 255) {
        class_hist[class_id]++;
      }
    }
  }
  
  uint64_t num_labels = 0;
  for (int i = 0; i < num_classes; i++)
    num_labels += class_hist[i];
  for (int i = 0; i < num_classes; i++) {
    if (class_hist[i] > 0)
      class_weights[i] = std::min(double(max_wgt), 1.0 / (double(class_hist[i]) / num_labels));
    else
      class_weights[i] = 0.0;
    //std::cout << class_hist[i] << '\n';
    //std::cout << class_weights[i] << '\n';
  }
  //#pragma omp parallel for
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int pos = r*width + c;
      uint8_t cidx = id_labels[pos];
      if (cidx < 255)
        weights[pos] = class_weights[cidx];
    }
  }
}
