#include "main.hpp"

#include <thread>
#include <vector>
#include <stdlib.h>

#include "set"
#include "iostream"

#define TEST_SIZE 64
// #define TEST

#define MODE_INFERENCE 0
#define MODE_TRAIN 1

src *src0;
sink *sink0;
dense *dense1, *dense2;
relu *relu1;
softmax *softmax2;

list_t *nns;
node_t *src0_n, *dense1_n, *relu1_n, *dense2_n, *softmax2_n, *sink0_n;
std::set<node_t*> versions[6];

int i_count = 0, o_count = 0;

uint checksum_history[TEST_SIZE * 2][5][2];

const int sizes[] = {4, 4, 4, 3, 3, 3};

inline uint generate_checksum(float *arr, int size) {
    uint sum = 0;
    for(int i = 0; i < size; ++i)
        sum += *(uint*)(&arr[i]);
    sum = ~sum;
    sum += 1;
    return sum;
}

template<class T>
inline int input(int mode, int thread_idx, rlu_thread_data_t *self, T *element, node_t *element_n, val_t data) {
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
inline int proceed(int mode, int thread_idx, rlu_thread_data_t *self, T *element, node_t *element_n, int index) {
restart:
    if (element_n == NULL) {
		perror("NULL from RLU_DEREF");
		exit(1);
	}
    if (!RLU_TRY_LOCK(self, &element_n)) {
		RLU_ABORT(self);
		return 0;
	}
    node_t* i = (node_t*)RLU_DEREF(self, (element_n));
    float *from = ((node_t*)RLU_DEREF(self, (element_n->p_prev)))->val;
    float *to = i->val;
    element->forward(from, to);
#ifdef TEST
    checksum_history[mode * TEST_SIZE + thread_idx][index][0] = generate_checksum(from, sizes[index]);
    checksum_history[mode * TEST_SIZE + thread_idx][index][1] = generate_checksum(to, sizes[index + 1]);
#endif
    versions[index+1].insert(i);
    return 1;
}

template<class T>
inline int output(int mode, int thread_idx, rlu_thread_data_t *self, T *element, node_t *element_n, val_t data) {
    if (element_n == NULL) {
		perror("NULL from RLU_DEREF");
		exit(1);
	}
    if (!RLU_TRY_LOCK(self, &element_n)) {
		RLU_ABORT(self);
        return 0;
	}
    node_t* i = (node_t *)RLU_DEREF(self, element_n);
    float *from = ((node_t*)RLU_DEREF(self, (element_n->p_prev)))->val;
    float *to = i->val;
    element->forward(from, to);
    element->forward(from, data);
#ifdef TEST
    checksum_history[mode * TEST_SIZE + thread_idx][5][0] = generate_checksum(from, sizes[4]);
    checksum_history[mode * TEST_SIZE + thread_idx][5][1] = generate_checksum(to, sizes[5]);
#endif
    versions[5].insert(i);
    ++o_count;
    return 1;
}

void inference(int thread_idx, rlu_thread_data_t* self, float data[], float result[]) {
    printf("Start: I%d, I / O count: %d / %d, current threads: %d\n", thread_idx, i_count, o_count, std::thread::hardware_concurrency);
restart:
    RLU_READER_LOCK(self);
	if(!input<src>(MODE_INFERENCE, thread_idx, self, src0, src0_n, data))
        goto restart;
    if(!proceed<dense>(MODE_INFERENCE, thread_idx, self, dense1, dense1_n, 0))
        goto restart;
    if(!proceed<relu>(MODE_INFERENCE, thread_idx, self, relu1, relu1_n, 1))
        goto restart;
    if(!proceed<dense>(MODE_INFERENCE, thread_idx, self, dense2, dense2_n, 2))
        goto restart;
    if(!proceed<softmax>(MODE_INFERENCE, thread_idx, self, softmax2, softmax2_n, 3))
        goto restart;
    if(!output<sink>(MODE_INFERENCE, thread_idx, self, sink0, sink0_n, data))
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
	if(!input<src>(MODE_TRAIN, thread_idx, self, src0, src0_n, data))
        goto restart;
    if(!proceed<dense>(MODE_TRAIN, thread_idx, self, dense1, dense1_n, 0))
        goto restart;
    if(!proceed<relu>(MODE_TRAIN, thread_idx, self, relu1, relu1_n, 1))
        goto restart;
    if(!proceed<dense>(MODE_TRAIN, thread_idx, self, dense2, dense2_n, 2))
        goto restart;
    if(!proceed<softmax>(MODE_TRAIN, thread_idx, self, softmax2, softmax2_n, 3))
        goto restart;
    if(!output<sink>(MODE_TRAIN, thread_idx, self, sink0, sink0_n, data))
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

#ifdef TEST
    for(int i = 0; i < TEST_SIZE; ++i) {
        printf("I%d: ", i);
        for(int j = 0; j < 4; ++j)
            printf("%010u --> %010u, ", checksum_history[i][j][0], checksum_history[i][j][1]);
        printf("%010u --> %010u\n", checksum_history[i][5][0], checksum_history[i][5][1]);
    }
    for(int i = 0; i < TEST_SIZE; ++i) {
        printf("T%d: ", i);
        for(int j = 0; j < 4; ++j)
            printf("%010u --> %010u, ", checksum_history[TEST_SIZE + i][j][0], checksum_history[TEST_SIZE + i][j][1]);
        printf("%010u --> %010u\n", checksum_history[TEST_SIZE + i][5][0], checksum_history[TEST_SIZE + i][5][1]);
    }
#endif
    return 0;
}