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

namespace Eigen
{
	typedef Eigen::Map<Eigen::VectorXd> MapX;
	typedef Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> MapX_3;
}

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
	void memory_setting();
	void free_cuda();

	std::unique_ptr<double[]> points_data;
	std::unique_ptr<double[]> centers_data;
	std::unique_ptr<double[]> normal_data;
	std::unique_ptr<double[]> mu_data;
	std::unique_ptr<double[]> mu1_data;
	std::unique_ptr<double[]> Fdmu_data;
	std::unique_ptr<double[]> Gdmu_data;
	std::unique_ptr<double[]> Edn_data;
	std::unique_ptr<double[]> Edn_data_temp;

	std::vector<double> node_pos;
	std::vector<double> node_width;
	std::vector<int> ngbr_list;
	std::vector<int> ngbr_list_startid;
	std::vector<int> ngbr_size_list;

	// multiply kernel needed
	std::array<double*, 2> d_query_points;	// points or centers, pre-allocated
	//double* d_points;						// equal to d_query_datap[0]
	double* d_normals;						// normals
	std::array<double*, 2> d_in_attr;		// values for points or centers
	double* d_density;						// wds, pre-allocated
	double* d_node_pos;						// required nodes info, pre-allocated
	double* d_node_width;
	int* d_ngbr_list;
	int* d_ngbr_list_startid;
	int* d_ngbr_size_list;					

	std::array<double*, 2> d_out_attr;		// values
	double* d_out_derivatives;

	void forward_A(Eigen::MapX& mu, Eigen::MapX& mu1);
	void forward_AT(Eigen::MapX& Edn);
};


#endif // !OCTTREE_H
