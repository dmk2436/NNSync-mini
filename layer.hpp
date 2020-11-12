#define DELAY

#define EMPTY 0
#define DENSE 1
#define RELU 2
#define SOFTMAX 3
#define SRC 80
#define SINK 88

typedef unsigned int uint;

// float learning_rate = 0.001;

class layer {
    public:
    int type = EMPTY;

    float* gradient;

    layer();
    virtual ~layer();

    void forward(float* prev, float* next);
    void backward(float* next, float* prev, float* upper, float* lower);
};

class dense: layer {
    public:
    int type = DENSE;
    uint w_from, w_to;

    float* weights;
    float* bias;

    dense();
    dense(layer& prev, uint w_from, uint w_to);
    virtual ~dense();

    void forward(float* prev, float* next);
    void backward(float* next, float* prev, float* upper, float* lower);
};

class relu: layer {
    public:
    int type = RELU;
    uint w_size;

    relu();
    relu(layer& prev, uint w_size);
    virtual ~relu();

    void forward(float* prev, float* next);
    void backward(float* next, float* prev, float* upper, float* lower);
};

class softmax: layer {
    public:
    int type = SOFTMAX;
    uint w_size;
    
    softmax();
    softmax(layer& prev, uint w_size);
    virtual ~softmax();

    void forward(float* prev, float* next);
    void backward(float* next, float* prev, float* upper, float* lower);
};

class src: layer {
    public:
    int type = SRC;

    uint w_size;

    src();
    src(uint w_size);
    virtual ~src();

    void forward(float* prev, float* next);
    void backward(float* next, float* prev);
};

class sink: layer {
    public:
    int type = SINK;

    uint w_size;

    sink();
    sink(layer& prev, uint w_size);
    virtual ~sink();

    void forward(float* prev, float* next);
    void backward(float* next, float* prev);
};