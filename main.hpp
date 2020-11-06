#include "layer.hpp"
#include "node.hpp"

void inference(int thread_idx, rlu_thread_data_t* self, float data[], float result[]);
void train(int thread_idx, rlu_thread_data_t* self, float data[], float result[]);

int main();