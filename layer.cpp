#include "layer.hpp"

#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#define coordinate(x, y, w, h) (w) * (y) + (x)

layer::layer() {}

layer::~layer() {}

dense::dense(layer& prev, uint w_from, uint w_to) {
    this->w_from = w_from;
    this->w_to = w_to;

    weights = (float*)malloc(this->w_from * this->w_to * sizeof(float));
    bias = (float*)malloc(this->w_to * sizeof(float));

    for(int i = 0; i < this->w_from; ++i)
        for(int j = 0; j < this->w_to; ++j)
            weights[coordinate(i, j, this->w_from, this->w_to)] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < this->w_to; ++i)
        bias[i] = (float)rand() / (float)RAND_MAX;
}

dense::~dense() {}

void dense::forward(float* prev, float* next) {
    for(int i = 0; i < w_to; ++i)
        next[i] = 0.0;
    for(int i = 0; i < w_to; ++i)
        for(int j = 0; j < w_from; ++j)
            next[i] += prev[j] * weights[coordinate(j, i, w_from, w_to)];
    for(int i = 0; i < w_to; ++i)
        next[i] += bias[i];
    
#ifdef DELAY
    sleep(1);
#endif
}

void dense::backward(float* next, float* prev, float* upper, float* lower) {
    free(gradient);
    gradient = (float*)malloc(w_from * sizeof(float));

    for(int i = 0; i < w_from; ++i)
        gradient[i] = 0.0;

    for(int i = 0; i < w_from; ++i)
            for(int j = 0; j < w_to; ++j)
                gradient[i] += weights[coordinate(i, j, w_from, w_to)];
        
    for(int i = 0; i < w_from; ++i)
        lower[i] = upper[i] * gradient[i];
        
    float* grad = (float*)malloc(w_from * w_to * sizeof(float));
        
    for(int i = 0; i < w_to; ++i)
        for(int j = 0; j < w_from; ++j) {
            grad[coordinate(j, i, w_from, w_to)] = prev[j];
            grad[coordinate(j, i, w_from, w_to)] *= upper[j];
        }
    
    for(int i = 0; i < w_from; ++i)
        for(int j = 0; j < w_to; ++j)
            weights[coordinate(i, j, w_from, w_to)] -= 0.01 * grad[coordinate(i, j, w_from, w_to)];
    
    free(grad);

#ifdef DELAY
    sleep(5);
#endif
}

relu::relu(layer& prev, uint w_size) {
    this->w_size = w_size;
}

relu::~relu() {}

void relu::forward(float* prev, float* next) {
    for(int i = 0; i < w_size; ++i) {
        if(prev[i] >= 0)
            next[i] = prev[i];
        else
            next[i] = 0.0;
    }

#ifdef DELAY
    sleep(1);
#endif
}

void relu::backward(float* next, float* prev, float* upper, float* lower) {
    free(gradient);
    gradient = (float*)malloc(w_size * sizeof(float));
        
    for(int i = 0; i < w_size; ++i) {
        if(prev[i] >= 0)
            gradient[i] = 1;
        else
            gradient[i] = 0;
    }

    for(int i = 0; i < w_size; ++i)
        lower[i] = upper[i] * gradient[i];
        
#ifdef DELAY
        sleep(5);
#endif
}

softmax::softmax(layer& prev, uint w_size) {
    this->w_size = w_size;
}

softmax::~softmax() {}

void softmax::forward(float* prev, float* next) {
    float sum = 0.0;

    for(int i = 0; i < w_size; ++i)
        sum += expf(prev[i]);
        
    for(int i = 0; i < w_size; ++i)
        next[i] = expf(prev[i]) / sum;

#ifdef DELAY
    sleep(1);
#endif
}

void softmax::backward(float* next, float* prev, float* upper, float* lower) {
    free(gradient);
    gradient = (float*)malloc(w_size * sizeof(float));

    float sum = 0.0;
    for(int i = 0; i < w_size; ++i)
        sum += expf(prev[i]);
        
    for(int i = 0; i < w_size; ++i)
        gradient[i] = expf(prev[i]) / sum * (1.0 - expf(prev[i] / sum));
        
    for(int i = 0; i < w_size; ++i)
        lower[i] = upper[i] * gradient[i];

#ifdef DELAY
    sleep(5);
#endif
}

src::src(uint w_size) {
    this->w_size = w_size;
}

src::~src() {}

void src::forward(float* prev, float* next) {
    for(int i = 0; i < w_size; ++i)
        next[i] = prev[i];
}

void src::backward(float* next, float* prev) {
}

sink::sink(layer& prev, uint w_size) {
    this->w_size = w_size;
}

sink::~sink() {}

void sink::forward(float* prev, float* next) {
    for(int i = 0; i < w_size; ++i)
        next[i] = prev[i];
}

void sink::backward(float* next, float* prev) {
}