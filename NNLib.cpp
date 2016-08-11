#include "NNLib.h"
#include <iostream>

VectorMatrix::VectorMatrix(){
	head = tail = NULL;
	len = 0;
}

VectorMatrix::~VectorMatrix(){
	deleteVec();
}

void VectorMatrix::deleteVec(){
	Node *curr = head;

	while (curr != NULL){
		Node *tmp = curr;
		curr = curr->next;

		delete tmp;
	}

	len = 0;
}

void VectorMatrix::createNode(int numNode){
	deleteVec();

	head = new Node;
	head->next = tail;
	Node *curr = head;

	for (int i = 0; i < numNode - 1; i++){
		curr->next = new Node;
		curr = curr->next;
	}

	curr->next = NULL;
	tail = curr;
	len = numNode;
}

Matrix& VectorMatrix::operator[](int pos){
	Node *curr = head;

	for (int i = 0; i < pos; i++){
		curr = curr->next;
	}

	return curr->data;
}