// Author: Johannes L. Schoenberger

#include "sift_matcher.h"

int main(int argc, char **argv) {
  // Match two images
  SIFTMatcher matcher;
  matcher.print_device_info();

  FeatureDescriptors desc1(100, 128);
  FeatureDescriptors desc2(20, 128);

  desc1.setRandom(29, 128);
  desc2.setRandom(10, 128);

  matcher.upload_descriptors(1, desc1);
  matcher.upload_descriptors(2, desc2);

  std::cout << matcher.match(1, 2) << std::endl;

  // Match multiple images by only uploading the descriptors to GPU once
  matcher = SIFTMatcher(0, 4);

  FeatureDescriptors desc3(120, 128);
  FeatureDescriptors desc4(40, 128);

  desc3.setRandom(120, 128);
  desc4.setRandom(40, 128);

  matcher.upload_descriptors(1, desc1);
  matcher.upload_descriptors(2, desc2);
  matcher.upload_descriptors(3, desc3);
  matcher.upload_descriptors(4, desc4);

  // Exhaustively match all image pairs
  for (size_t i = 1; i < 5; ++i) {
    for (size_t j = 1; j < i; ++j) {
      std::cout << matcher.match(i, j) << std::endl;
    }
  }

  return EXIT_SUCCESS;
}
