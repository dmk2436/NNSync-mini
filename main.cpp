#include "main.hpp"

#include <thread>
#include <vector>
#include <stdlib.h>

#include "set"
#include "iostream"

#define TEST_SIZE 64

src *src0;
sink *sink0;
dense *dense1, *dense2;
relu *relu1;
softmax *softmax2;

list_t *nns;
node_t *src0_n, *dense1_n, *relu1_n, *dense2_n, *softmax2_n, *sink0_n;
std::set<node_t*> versions[6];

int i_count = 0, o_count = 0;

template<class T>
int proceed(rlu_thread_data_t *self, T *element, node_t *element_n, int index) {
restart:
    if (element_n == NULL) {
		perror("NULL from RLU_DEREF");
		exit(1);
	}
    if (!RLU_TRY_LOCK(self, &element_n)) {
		RLU_ABORT(self);
		return 0;
	}
    node_t* i = (node_t*)RLU_DEREF(self, (element_n->p_prev));
    element->forward(i->val, ((node_t*)RLU_DEREF(self, (element_n)))->val);
    versions[index+1].insert(i);
    return 1;
}

template<class T>
int input(rlu_thread_data_t *self, T *element, node_t *element_n, val_t data) {
    if (element_n == NULL) {
		perror("NULL from RLU_DEREF");
		exit(1);
	}
    if (!RLU_TRY_LOCK(self, &element_n)) {
		RLU_ABORT(self);
        return 0;
	}
    node_t* i = (node_t *)RLU_DEREF(self, (element_n));
    element->forward(data, i->val);
    versions[0].insert(i);
    ++i_count;
    return 1;
}

template<class T>
int output(rlu_thread_data_t *self, T *element, node_t *element_n, val_t data) {
    if (element_n == NULL) {
		perror("NULL from RLU_DEREF");
		exit(1);
	}
    if (!RLU_TRY_LOCK(self, &element_n)) {
		RLU_ABORT(self);
        return 0;
	}
    node_t* i = (node_t *)RLU_DEREF(self, element_n);
    element->forward(((node_t*)RLU_DEREF(self, (element_n->p_prev)))->val, i->val);
    element->forward(((node_t*)RLU_DEREF(self, (element_n->p_prev)))->val, data);
    versions[5].insert(i);
    ++o_count;
    return 1;
}

void inference(int thread_idx, rlu_thread_data_t* self, float data[], float result[]) {
    printf("Start: I%d, I / O count: %d / %d, current threads: %d\n", thread_idx, i_count, o_count, std::thread::hardware_concurrency);
restart:
    RLU_READER_LOCK(self);
	if(!input<src>(self, src0, src0_n, data))
        goto restart;
    if(!proceed<dense>(self, dense1, dense1_n, 0))
        goto restart;
    if(!proceed<relu>(self, relu1, relu1_n, 1))
        goto restart;
    if(!proceed<dense>(self, dense2, dense2_n, 2))
        goto restart;
    if(!proceed<softmax>(self, softmax2, softmax2_n, 3))
        goto restart;
    if(!output<sink>(self, sink0, sink0_n, data))
        goto restart;
    std::string output = "Versions: ";
    for(int i = 0; i < 5; i++)
        output.append(std::to_string(versions[i].size()) + " --> ");
    output.append(std::to_string(versions[5].size()));
    std::cout<<output<<std::endl;
    RLU_READER_UNLOCK(self);
    printf("End: I%d, I / O count: %d / %d, current threads: %d\n", thread_idx, i_count, o_count, std::thread::hardware_concurrency());
}

void train(int thread_idx, rlu_thread_data_t* self, float data[], float result[]) {
    printf("Start: T%d, I / O count: %d / %d, current threads: %d\n", thread_idx, i_count, o_count, std::thread::hardware_concurrency);
restart:
    RLU_READER_LOCK(self);
    // forward
	if(!input<src>(self, src0, src0_n, data))
        goto restart;
    if(!proceed<dense>(self, dense1, dense1_n, 0))
        goto restart;
    if(!proceed<relu>(self, relu1, relu1_n, 1))
        goto restart;
    if(!proceed<dense>(self, dense2, dense2_n, 2))
        goto restart;
    if(!proceed<softmax>(self, softmax2, softmax2_n, 3))
        goto restart;
    if(!output<sink>(self, sink0, sink0_n, data))
        goto restart;

    // backward
    float init_grad[3] = {1.0, 1.0, 1.0};
    float softmax2_grad[3];
    softmax2->backward(((node_t*)RLU_DEREF(self, (softmax2_n)))->val, ((node_t*)RLU_DEREF(self, (sink0_n)))->val, init_grad, softmax2_grad);

    float dense2_grad[3 * 4];
    dense2->backward(((node_t*)RLU_DEREF(self, (dense2_n)))->val, ((node_t*)RLU_DEREF(self, (softmax2_n)))->val, softmax2_grad, dense2_grad);
    
    float relu1_grad[4];
    relu1->backward(((node_t*)RLU_DEREF(self, (relu1_n)))->val, ((node_t*)RLU_DEREF(self, (dense2_n)))->val, dense2_grad, relu1_grad);

    float dense1_grad[4 * 4];
    dense1->backward(((node_t*)RLU_DEREF(self, (dense1_n)))->val, ((node_t*)RLU_DEREF(self, (relu1_n)))->val, relu1_grad, dense1_grad);

    std::string output = "Versions: ";
    for(int i = 0; i < 5; i++)
        output.append(std::to_string(versions[i].size()) + " --> ");
    output.append(std::to_string(versions[5].size()));
    std::cout<<output<<std::endl;
    RLU_READER_UNLOCK(self);
    printf("End: T%d, I / O count: %d / %d, current threads: %d\n", thread_idx, i_count, o_count, std::thread::hardware_concurrency());
}

int main() {
    RLU_INIT();

    auto self = RLU_THREAD_ALLOC();
    RLU_THREAD_INIT(self);

    nns = rlu_new_list();

    src0 = new src(4);
    dense1 = new dense((layer&)src0, 4, 4);
    relu1 = new relu((layer&)dense1, 4);
    dense2 = new dense((layer&)relu1, 4, 3);
    softmax2 = new softmax((layer&)dense2, 3);
    sink0 = new sink((layer&)softmax2, 3);

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

    float data[TEST_SIZE][4];
    float result[TEST_SIZE][3];
    for(int i = 0; i < TEST_SIZE; ++i) {
        for(int j = 0; j < 4; ++j)
            data[i][j] = (float)rand() / (float)RAND_MAX;
        for(int j = 0; j < 3; ++j)
            result[i][j] = 0.0;
    }

    std::vector<std::thread*> t;
    for(int i = 0; i < TEST_SIZE; ++i) {
        auto _ti = new std::thread(inference, i, self, data[i], result[i]);
        t.push_back(_ti);
        auto _tt = new std::thread(train, i, self, data[i], result[i]);
        t.push_back(_tt);
    }
    for (auto it : t)
        it->join();
    
    RLU_THREAD_FINISH(self);
    RLU_THREAD_FREE(self);

    RLU_FINISH();
    return 0;
}