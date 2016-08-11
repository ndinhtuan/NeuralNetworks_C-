#ifndef NN
#define NN
#include "Matrix.h"
#include <vector>

using std::vector;

#include <iostream>
#include "NNLib.h"

class NeuralNetworks
{
public:
	NeuralNetworks();
	~NeuralNetworks();
	void loadData(const char *srcFile);
	void loadX(const char *srcFile);
	void loadY(const char *srcFile);
	void loadTheta(const char *srcFile, Matrix &thetaI);

	bool loadedData();
	bool inited_Layer();
	void setNumLayers(int num){
		numLayer = num;
		theta.createNode(num);
		grad.createNode(num);
	}
	Matrix& getThetaI(int i){
		return theta[i];
	}
	Matrix& getY(){
		return y;
	}
	void setNumOutputs(int numOuts){
		numOutputs = numOuts;
	}
	double trainAcurrateNN();
	Matrix randInitTheta(int Lin, int Lout);
	void computeJreg(double lambda);
	double getJreg(){
		return Jreg;
	}
	void updateGrad(double lambda);
	Matrix& getGradI(int i){
		return grad[i];
	}
private:
	VectorMatrix theta;
	VectorMatrix grad;
	Matrix X;
	Matrix y;
	int mX;
	int nX;
	int numLayer;
	int numOutputs;
	double Jreg;
};
#endif

