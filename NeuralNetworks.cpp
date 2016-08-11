#include "NeuralNetworks.h"
#include <fstream>

using std::ifstream;
using std::ofstream;

#include <iostream>

using std::cin;
using std::cout;
using std::endl;

#include "MatrixLib.h"
#include <conio.h>
#include <sstream>

using std::stringstream;

#include <string.h>

using std::string;

#include <cassert>

NeuralNetworks::NeuralNetworks()
{
	numLayer = 0;
	mX = nX = 0;
	numOutputs = 0;
}


NeuralNetworks::~NeuralNetworks()
{
}

void NeuralNetworks::loadData(const char *srcFile){
	ifstream input(srcFile);

	if (!input.is_open()){
		cout << "Cannot access file to read data for X and y." << endl;
		return;
	}

	input >> mX >> nX;

	X.createMat(mX, nX);
	y.createMat(mX, 1);

	for (int row = 0; row < mX; row++){

		for (int col = 0; col < nX; col++){
			input >> X[row][col];
		}

		input >> y[row][0];
	}
}
void NeuralNetworks::loadX(const char *srcFile){
	ifstream input(srcFile);

	if (!input.is_open()){
		cout << "Cannot access file " << srcFile << " to read data for X." << endl;
		return;
	}

	input >> mX >> nX;

	X.createMat(mX, nX);

	for (int row = 0; row < mX; row++){

		for (int col = 0; col < nX; col++){
			if (!input.eof()){
				if ((int)input.tellg() != -1) input >> X[row][col];
			}
			else{
				cout << "End of file  " << srcFile << " " << row << "; " << col << endl;
				_getch();
			}
		}

	}
}
void NeuralNetworks::loadY(const char *srcFile){
	y.createMat(mX, 1);
	ifstream input(srcFile);

	if (!input.is_open()){
		cout << "Cannot access file " << srcFile << " to read data for y. " << endl;
		return;
	}

	input >> mX;

	for (int row = 0; row < mX; row++){
		if (!input.eof()){
			if ((int)input.tellg() != -1) input >> y[row][0];
		}
		else{
			cout << "End of file  " << srcFile << " " << row << "; " << 0 << endl;
			_getch();
		}
	}
}

void NeuralNetworks::loadTheta(const char *srcFile, Matrix &Thetai){
	ifstream input(srcFile);

	if (!input.is_open()){
		cout << "Cannot access file " << srcFile << " to read data for Theta." << endl;
		return;
	}

	input >> Thetai.getSize();
	Thetai.createMat(Thetai.getSize().rows, Thetai.getSize().cols);

	for (int row = 0; row < Thetai.getSize().rows; row++){

		for (int col = 0; col < Thetai.getSize().cols; col++){
			
			if (!input.eof()){
				if((int)input.tellg() != -1) input >> Thetai[row][col];
			}
			else{
				cout << "End of file  " << srcFile << " " << row << "; " << col << endl;
				_getch();
			}

		}

	}

}

bool NeuralNetworks::loadedData(){
	return (mX != 0) && (nX != 0);
}

bool NeuralNetworks::inited_Layer(){
	return numLayer != 0;
}

double NeuralNetworks::trainAcurrateNN(){
	if (!loadedData() || !inited_Layer()){
		cout << "Load data and init layers before run forward propagation." << endl;
		return 0;
	}

	Matrix _y;
	Matrix z, a;
	a = X;

	for (int layer = 0; layer < numLayer - 1; layer++){
		a.addX0();
		z = a * theta[layer].transpose();
		a = sigmoid(z);
/*
		stringstream ss;
		string nameFile;

		ss << 'z' << layer << ".txt";
		std::getline(ss, nameFile);
		ofstream out1(nameFile.c_str());
		out1 << z;

		ss.clear();
		ss << 'a' << layer << ".txt";
		std::getline(ss, nameFile);
		ofstream out(nameFile.c_str());
		out << a;
		ss.clear();*/
	}

	_y = a.maxInRows().elementsOfCol(1);
	_y = _y + 1;

	if (y.getSize().rows != _y.getSize().rows){
		cout << "y 's rows - " << y.getSize().rows << " should equal _y 's rows.. - " << _y.getSize().rows << endl;
		return 0;
	}

	int truePredict = mX;

	for (int row = 0; row < mX; row++){
		if (_y[row][0] != y[row][0] && (_y[row][0] != 0 && y[row][0] != 10)){
			truePredict--;
		}
	}
	ofstream out("yNN.txt");
	out << _y;
	return double(truePredict) * 100 / mX;
}

Matrix NeuralNetworks::randInitTheta(int Lin, int Lout){
	Matrix result;

	double epsilon = sqrt(6) / sqrt(Lout + Lin);
	result = rand(Lout, Lin + 1) * 2 * epsilon - epsilon;

	return result;
}

void NeuralNetworks::computeJreg(double lambda){
	if (!loadedData()){
		cout << "Need to load Data before computing cost function." << endl;
		return;
	}

	assert(numOutputs != 0);

	if (numOutputs == 2){
		// Do something
		return;
	}

	Matrix _y(y.getSize().rows, numOutputs, 0);
	for (int row = 0; row < y.getSize().rows; row++){
		_y[row][ (int)y[row][0] - 1] = 1;
	}
	
	Matrix a = X;
	Matrix z;

	for (int i = 0; i < numLayer - 1; i++){
		a.addX0();
		z = a * theta[i].transpose();
		a = sigmoid(z);
	}

	Matrix tmp = _y.multiEachElement(log(a)) + (1 - _y).multiEachElement(log(1 - a));
	Jreg = (double(-1) / mX) * tmp.transpose().sum().transpose().sum()[0][0];
	
	for (int layer = 0; layer < numLayer - 1; layer++){

		for (int row = 0; row < theta[layer].getSize().rows; row++){

			for (int col = 0; col < theta[layer].getSize().cols; col++){

				if (col != 0){
					Jreg += (lambda / (2 * mX)) * pow(theta[layer][row][col], 2);
				}

			}

	}
}
}

void NeuralNetworks::updateGrad(double lambda){

	if (!loadedData()){
		cout << "Need to load Data before computing cost function." << endl;
		return;
	}

	assert(numOutputs != 0);

	if (numOutputs == 2){
		// Do something
		return;
	}

	Matrix _y(y.getSize().rows, numOutputs, 0);
	for (int row = 0; row < y.getSize().rows; row++){
		_y[row][(int)y[row][0] - 1] = 1;
	}

	for (int t = 0; t < X.getSize().rows; t++){
		VectorMatrix a;
		a.createNode(numLayer);
		a[0] = X.elementsOfRow(t);
		Matrix z;

		for (int i = 0; i < numLayer - 1; i++){
			a[i].addX0();
			z = a[i] * theta[i].transpose();
			a[i + 1] = sigmoid(z);
		}

		VectorMatrix delta;
		delta.createNode(numLayer);
		delta[numLayer - 1] = a[numLayer - 1];

		for (int i = numLayer - 2; i > 0; i--){ // delta[0] no need to compute
			delta[i] = theta[i].transpose() * delta[i + 1] * a[i] * (1 - a[i]);
		}

		for (int i = 0; i < numLayer - 1; i++){
			grad[i] = grad[i] + delta[i + 1] * (a[i].transpose());
		}
	}

	for (int i = 0; i < numLayer - 1; i++){
		grad[i] = (grad[i] + theta[i].changeValueInCol(0, 0) * lambda) * (double (1) / mX);
	}
}
