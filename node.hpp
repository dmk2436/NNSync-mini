#include "mvrlu.h"

#include "types.hpp"

node_t* rlu_new_node();
list_t* rlu_new_list();

int rlu_list_size(rlu_thread_data_t* self, list_t* p_list);
int rlu_list_add(rlu_thread_data_t* self, list_t* p_list, node_t* p_new_node, val_t val);
int rlu_list_add(rlu_thread_data_t* self, node_t* p_prev, node_t* p_new_node, val_t val);
int rlu_node_remove(rlu_thread_data_t* self, node_t* p_node);
int rlu_list_remove(rlu_thread_data_t* self, list_t* p_list, val_t val);