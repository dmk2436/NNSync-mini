typedef float* val_t;

typedef struct node {
    val_t val;
    struct node* p_next;
} node_t;

typedef struct list {
    node_t* p_head;
} list_t;