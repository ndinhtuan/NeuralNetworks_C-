#ifndef NNLIB
#define NNLIB
#include "Matrix.h"

struct Node{
	Matrix data;
	Node *next;
};

class VectorMatrix
{
public:
	VectorMatrix();
	~VectorMatrix();
	void createNode(int numNode);
	Matrix& operator[](int i);
	void deleteVec();
private:
	Node *head;
	Node *tail;
	int len;
};
#endif

