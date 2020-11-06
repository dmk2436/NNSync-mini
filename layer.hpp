#define DELAY

#define EMPTY 0
#define DENSE 1
#define RELU 2
#define SOFTMAX 3

typedef unsigned int uint;

// float learning_rate = 0.001;

class layer {
    public:
    int type = EMPTY;

    float* gradient;

    layer();
    virtual ~layer();

    void forward(float* prev, float* dest);
    void backward(float* prev, float* dest, float* upper, float* lower);
};

class dense: layer {
    public:
    uint w_from, w_to;

    float* weights;
    float* bias;

    dense();
    dense(layer& prev, uint w_from, uint w_to);
    virtual ~dense();

    void forward(float* prev, float* dest);
    void backward(float* prev, float* dest, float* upper, float* lower);
};

class relu: layer {
    public:
    uint w_size;

    relu();
    relu(layer& prev, uint w_size);
    virtual ~relu();

    void forward(float* prev, float* dest);
    void backward(float* prev, float* dest, float* upper, float* lower);
};

class softmax: layer {
    public:
    uint w_size;
    
    softmax();
    softmax(layer& prev, uint w_size);
    virtual ~softmax();

    void forward(float* prev, float* dest);
    void backward(float* prev, float* dest, float* upper, float* lower);
};