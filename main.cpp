#include "main.hpp"

#include <thread>
#include <vector>
#include <stdlib.h>

void arraycpy(float* dest, const float *src, int cnt) {
    for(int i = 0; i < cnt; ++i)
        dest[i] = src[i];
}

layer *src0;
dense *dense1, *dense2;
relu *relu1;
softmax *softmax2;

list_t *nns;
node_t *src0_n, *dense1_n, *relu1_n, *dense2_n, *softmax2_n, *sink0_n;

void inference(int thread_idx, rlu_thread_data_t* self, float data[], float result[]) {
restart:
    RLU_READER_LOCK(self);
	if (!RLU_TRY_LOCK(self, &src0_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    arraycpy(src0_n->val, data, 4);

    if (!RLU_TRY_LOCK(self, &dense1_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    dense1->forward(src0_n->val, dense1_n->val);

    if (!RLU_TRY_LOCK(self, &relu1_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    relu1->forward(dense1_n->val, relu1_n->val);

    if (!RLU_TRY_LOCK(self, &dense2_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    dense2->forward(relu1_n->val, dense2_n->val);

    if (!RLU_TRY_LOCK(self, &softmax2_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    softmax2->forward(dense2_n->val, softmax2_n->val);

    arraycpy(result, softmax2_n->val, 3);
    RLU_READER_UNLOCK(self);
}

void train(int thread_idx, rlu_thread_data_t* self, float data[], float result[]) {
restart:
    RLU_READER_LOCK(self);
    // forward
	if (!RLU_TRY_LOCK(self, &src0_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    arraycpy(src0_n->val, data, 4);

    if (!RLU_TRY_LOCK(self, &dense1_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    dense1->forward(src0_n->val, dense1_n->val);

    if (!RLU_TRY_LOCK(self, &relu1_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    relu1->forward(dense1_n->val, relu1_n->val);

    if (!RLU_TRY_LOCK(self, &dense2_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    dense2->forward(relu1_n->val, dense2_n->val);

    if (!RLU_TRY_LOCK(self, &softmax2_n)) {
		RLU_ABORT(self);
		goto restart;
	}
    softmax2->forward(dense2_n->val, softmax2_n->val);

    arraycpy(result, softmax2_n->val, 3);

    // backward
    float init_grad[3] = {1.0, 1.0, 1.0};
    float softmax2_grad[3] = {0.0, };
    softmax2->backward(softmax2_n->val, sink0_n->val, init_grad, softmax2_grad);

    float dense2_grad[3 * 4] = {0.0, };
    dense2->backward(dense2_n->val, softmax2_n->val, softmax2_grad, dense2_grad);

    float relu1_grad[4] = {0.0, };
    relu1->backward(relu1_n->val, dense2_n->val, dense2_grad, relu1_grad);

    float dense1_grad[4 * 4] = {0.0, };
    dense1->backward(dense1_n->val, relu1_n->val, relu1_grad, dense1_grad);
    RLU_READER_UNLOCK(self);
}

int main() {
    RLU_INIT();

    auto self = RLU_THREAD_ALLOC();
    RLU_THREAD_INIT(self);

    nns = rlu_new_list();

    src0 = new layer();
    dense1 = new dense((layer&)src0, 4, 4);
    relu1 = new relu((layer&)dense1, 4);
    dense2 = new dense((layer&)relu1, 4, 3);
    softmax2 = new softmax((layer&)dense2, 3);

    src0_n = rlu_new_node();
    float src0_p[4] = {0.0, };
    rlu_list_add(self, nns, src0_n, src0_p);

    dense1_n = rlu_new_node();
    float dense1_p[4] = {0.0, };
    rlu_list_add(self, src0_n, dense1_n, dense1_p);

    relu1_n = rlu_new_node();
    float relu1_p[4] = {0.0, };
    rlu_list_add(self, dense1_n, relu1_n, relu1_p);

    dense2_n = rlu_new_node();
    float dense2_p[3] = {0.0, };
    rlu_list_add(self, relu1_n, dense2_n, dense2_p);

    softmax2_n = rlu_new_node();
    float softmax2_p[3] = {0.0, };
    rlu_list_add(self, dense2_n, softmax2_n, softmax2_p);

    sink0_n = rlu_new_node();
    float sink0_p[3] = {0.0, };
    rlu_list_add(self, softmax2_n, sink0_n, sink0_p);

    float data[100][4];
    float result[100][3];
    for(int i = 0; i < 100; ++i) {
        for(int j = 0; j < 4; ++j)
            data[i][j] = (float)rand() / (float)RAND_MAX;
        for(int j = 0; j < 3; ++j)
            result[i][j] = 0.0;
    }

    std::vector<std::thread*> t;
    for(int i = 0; i < 1; ++i) {
        auto _t = new std::thread(inference, i, self, data[i], result[i]);
        t.push_back(_t);
    }
    for (auto it : t)
        it->join();
    
    RLU_THREAD_FINISH(self);
    RLU_THREAD_FREE(self);

    RLU_FINISH();
    return 0;
}