#pragma once
#ifndef OCTTREE_H
#define OCTTREE_H
#define EIGEN_USE_MKL_ALL
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>

#include <cuda.h>
#include <cuda_runtime.h>

#include "basicStructure.h"
#include "pointCloud.h"
#include "octNode.h"
#include "HLBFGS/HLBFGS.h"
#include "ANN/ANN.h"

class OctTree
{
	typedef Eigen::Triplet<double> Td;
	typedef std::pair<int, double> pa;
public:
	OctTree(PointCloud *p);
	std::vector<OctNode*> nodeArr;
	std::vector<OctNode*> leafNodeArr;
	std::vector<Point> points;
	std::vector<int> pointNodeId;
	std::vector<int> nopNode;
	std::vector<int> outNode;
	std::vector<int> depthNode[MAXDEPTH];
	std::vector<std::vector<double>> near_points;
	std::vector<std::vector<int>> near_points_id;
	std::vector<std::vector<int>> Ngbr;
	std::vector<double> wds;
	std::vector<double> avgdis; 
	std::vector<int> pdep;

	Eigen::SparseMatrix<double> A;
	Eigen::SparseMatrix<double> B[3];
	Eigen::SparseMatrix<double> B_T[3];
	Eigen::SparseMatrix<double> P;
	Eigen::SparseMatrix<double> P1;
	Eigen::SparseMatrix<double> P_T;
	Eigen::SparseMatrix<double> P1_T;
	Eigen::SparseMatrix<double> P2;

	Eigen::PardisoLLT <Eigen::SparseMatrix<double>> A_solver;
	void energy_evaluation_w_uv(const std::vector<double>& x, double& f, std::vector<double>& g);
	void energy_evaluation_w_print_p(const std::vector<double>& x, std::vector<double>& p);
	void optimize_LBFS_w_uv(std::vector<double>& x);
	void clear_near_points();
	ANNkd_tree* kdTree = NULL;
	ANNkd_tree* kdTree_o = NULL;
	ANNkd_tree* kdTree_no = NULL;
	void set_gamma(double gam);

private:
	double gamma = 100000;
	double sig = 0.002;
	double pxMax;
	double pxMin;
	double pyMax;
	double pyMin;
	double pzMax;
	double pzMin;
	PointCloud *pc;
	void unitPointCloud();
	void buildTree();
	void find_around();
	void calc_B();
	void calc_A();
	void calc_P();
	void find_near_points();
	void find_Ngbr();
	void calc_avgdis();
	void get_knear(Point p, int k, std::vector<int> &pid);
	void get_allpoints(std::vector<int> &pid, int nid);
	void get_outNode();

	void serialize_tree();
	std::unique_ptr<double[]> normal_data;
	std::unique_ptr<double[]> mu_data;
	std::unique_ptr<double[]> mu1_data;
	std::unique_ptr<double[]> Fdmu_data;
	std::unique_ptr<double[]> Gdmu_data;

	std::array<std::vector<double>, 2> node_pos;
	std::array<std::vector<double>, 2 > node_width;
	std::array<std::vector<int>, 2 > ngbr_list;
	std::array<std::vector<int>, 2 > ngbr_list_startid;
	std::array<std::vector<int>, 2 > ngbr_size_list;

	std::array<double*, 2> d_node_pos;
	std::array<double*, 2 > d_node_width;
	std::array<int*, 2 > d_ngbr_list;
	std::array<int*, 2 > d_ngbr_list_startid;
	std::array<int*, 2 > d_ngbr_size_list;

	void free_cuda();

	void forward_A();
	void forward_AT();
};


#endif // !OCTTREE_H
