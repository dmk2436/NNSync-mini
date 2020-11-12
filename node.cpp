#include "node.hpp"

#include <stdio.h>
#include <stdlib.h>

node_t* rlu_new_node() {
    node_t* p_new_node = (node_t*)RLU_ALLOC(sizeof(node_t));
    if(p_new_node == NULL) {
        printf("ERROR: out of memory\n");
        exit(1);
    }
    return p_new_node;
}

list_t* rlu_new_list() {
    list_t* p_list;

    p_list = (list_t*)RLU_ALLOC(sizeof(list_t));
    if(p_list == NULL) {
        perror("malloc");
        exit(1);
    }

    node_t* p_head_node = rlu_new_node();
    p_list->p_head = p_head_node;

    return p_list;
}

int rlu_list_size(rlu_thread_data_t* self, list_t* p_list) {
    int size = 0;
    node_t *p_next, *p_prev;

    p_prev = (node_t*)RLU_DEREF(self, p_list->p_head->p_next);
    p_next = (node_t*)RLU_DEREF(self, p_prev->p_next);

    while(p_next != NULL) {
        size++;
        p_prev = p_next;
        p_next = (node_t*)RLU_DEREF(self, p_prev->p_next);
    }

    return size;
}

int rlu_list_add(rlu_thread_data_t* self, list_t* p_list, node_t* p_new_node, val_t val) {
    p_new_node->val = val;
    
    RLU_ASSIGN_PTR(self, &(p_list->p_head), p_new_node);
    RLU_ASSIGN_PTR(self, &(p_new_node->p_prev), NULL);

    return 1;
}

int rlu_list_add(rlu_thread_data_t* self, node_t* p_prev, node_t* p_new_node, val_t val) {
    p_new_node->val = val;

restart:
    RLU_READER_LOCK(self);

    if (p_prev == NULL) {
		perror("NULL from RLU_DEREF");
		exit(1);
	}

    if(!RLU_TRY_LOCK(self, &p_prev)) {
        RLU_ABORT(self);
        goto restart;
    }
    
    RLU_ASSIGN_PTR(self, &(p_prev->p_next), p_new_node);
    RLU_ASSIGN_PTR(self, &(p_new_node->p_prev), p_prev);

    RLU_READER_UNLOCK(self);

    return 1;
}

int rlu_node_remove(rlu_thread_data_t* self, node_t* p_node) {
    RLU_DEREF(self, p_node);
}

int rlu_list_remove(rlu_thread_data_t* self, list_t* p_list, val_t val) {
    int result;
    node_t *p_prev, *p_next;
    node_t* n;
    val_t v;
    
restart:
    RLU_READER_LOCK(self);
    
    p_prev = (node_t*)RLU_DEREF(self, p_list->p_head);
    p_next = (node_t*)RLU_DEREF(self, p_prev->p_next);

    while(1) {
        // p_node = (node_t*)RLU_DEREF(self, p_next);

        v = p_next->val;

        if(v >= val)
            break;
        
        p_prev = p_next;
        p_next = (node_t*)RLU_DEREF(self, p_prev->p_next);
    }

    result = (v == val);
    
    if(result) {
        n = (node_t*)RLU_DEREF(self, p_next->p_next);

        if(!RLU_TRY_LOCK(self, &p_prev)) {
            RLU_ABORT(self);
            goto restart;
        }
        if(!RLU_TRY_LOCK_CONST(self, p_next)) {
            RLU_ABORT(self);
            goto restart;
        }

        RLU_ASSIGN_PTR(self, &(p_prev->p_next), n);
        RLU_FREE(self, p_next);

        RLU_READER_UNLOCK(self);

        return result;
    }

    RLU_READER_UNLOCK(self);

    return result;
}