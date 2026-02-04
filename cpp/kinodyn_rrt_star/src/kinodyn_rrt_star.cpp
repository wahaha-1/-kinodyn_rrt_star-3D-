#include "kinodyn_rrt_star/kinodyn_rrt_star.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <chrono>

double StateNode::calcOptimalTrajWithFullState_(DroneState_t start, DroneState_t end, double &optimal_T, Eigen::Matrix<double, 6, 3> &coeff)
{
	Eigen::Vector3d p0 = start.pos, v0 = start.vel, a0 = start.acc;
	Eigen::Vector3d pf = end.pos, vf = end.vel, af = end.acc;
	Eigen::Matrix<double, 6, 6> m;

	double c4 = -9.0 * a0.squaredNorm() - 9.0 * af.squaredNorm() + 6.0 * a0.transpose() * af;
	double c3 = (af.transpose() * (96.0 * v0 + 144.0 * vf) - a0.transpose() * (96.0 * vf + 144.0 * v0))(0);
	double c2 = -576.0 * v0.squaredNorm() - 1008.0 * v0.transpose() * vf - 576.0 * vf.squaredNorm() +
							-360.0 * (p0 - pf).transpose() * (a0 - af);
	double c1 = -2880.0 * (v0 + vf).transpose() * (p0 - pf);
	double c0 = -3600.0 * (p0 - pf).squaredNorm();

	m << 0.0, 0.0, 0.0, 0.0, 0.0, -c0,
			1.0, 0.0, 0.0, 0.0, 0.0, -c1,
			0.0, 1.0, 0.0, 0.0, 0.0, -c2,
			0.0, 0.0, 1.0, 0.0, 0.0, -c3,
			0.0, 0.0, 0.0, 1.0, 0.0, -c4,
			0.0, 0.0, 0.0, 0.0, 1.0, 0.0;

	double cost = StateNode::kErrorCost, optimal_cost = StateNode::kErrorCost;
	double T = -1.0;
	optimal_T = -1.0;

	Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> eigen_values;
	eigen_values = m.eigenvalues();

	for (int i = 0; i < 6; ++i)
	{
		T = std::real(eigen_values(i));
		double img = std::imag(eigen_values(i));
		double freq = 1.0 / T;

		if (T <= 0 || std::abs(img) >= 1e-16)
			continue;

		Eigen::Vector3d k_alpha, k_beta, k_gamma;
		k_gamma = freq * (3.0 * af - 9.0 * a0 + freq * (-36.0 * v0 - 24.0 * vf + freq * 60.0 * (pf - p0)));
		k_beta = freq * freq * (-24.0 * af + 36.0 * a0 + freq * (192.0 * v0 + 168.0 * vf + freq * 360.0 * (p0 - pf)));
		k_alpha = freq * freq * freq * (60.0 * (af - a0) + freq * (-360.0 * (v0 + vf) + freq * 720.0 * (pf - p0)));

		cost = T * (1.0 + k_gamma.squaredNorm() +
								T * ((k_beta.transpose() * k_gamma)(0) +
										 T * ((k_beta.squaredNorm() + (k_alpha.transpose() * k_gamma)(0)) / 3.0 +
													T * ((k_alpha.transpose() * k_beta)(0) / 4.0 +
															 T * k_alpha.squaredNorm() / 20.0))));
		if (cost <= optimal_cost)
		{
			optimal_cost = cost;
			optimal_T = T;
			coeff.row(3) = k_gamma / 6.0;
			coeff.row(4) = k_beta / 24.0;
			coeff.row(5) = k_alpha / 120.0;
		}
	}

	coeff.row(0) = p0;
	coeff.row(1) = v0;
	coeff.row(2) = a0 / 2.0;

	if (optimal_T == -1.0)
	{
		std::cerr << "StateNode::calcOptimalTrajWithFullState_: optimal_T cannot be found from point(" << p0(0) << ", "
							<< p0(1) << ", " << p0(2) << ") to point(" << pf(0) << ", " << pf(1) << ", " << pf(2) << ")" << std::endl;
	}

	return optimal_cost;
}

double StateNode::calcOptimalTrajWithPartialState_(DroneState_t start, Eigen::Vector3d pf, double &optimal_T, Eigen::Matrix<double, 6, 3> &coeff)
{
	Eigen::Vector3d p0 = start.pos, v0 = start.vel, a0 = start.acc;
	Eigen::Matrix<double, 6, 6> m;

	double c4 = -5.0 * a0.squaredNorm();
	double c3 = -40.0 * a0.transpose() * v0;
	double c2 = -60.0 * (v0.squaredNorm() + a0.transpose() * (p0 - pf));
	double c1 = -160.0 * v0.transpose() * (p0 - pf);
	double c0 = -100.0 * (p0 - pf).squaredNorm();

	m << 0.0, 0.0, 0.0, 0.0, 0.0, -c0,
			1.0, 0.0, 0.0, 0.0, 0.0, -c1,
			0.0, 1.0, 0.0, 0.0, 0.0, -c2,
			0.0, 0.0, 1.0, 0.0, 0.0, -c3,
			0.0, 0.0, 0.0, 1.0, 0.0, -c4,
			0.0, 0.0, 0.0, 0.0, 1.0, 0.0;

	double cost = StateNode::kErrorCost, optimal_cost = StateNode::kErrorCost;
	double T = -1.0;
	optimal_T = -1.0;

	Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> eigen_values;
	eigen_values = m.eigenvalues();

	for (int i = 0; i < 6; ++i)
	{
		T = std::real(eigen_values(i));
		double img = std::imag(eigen_values(i));
		double freq = 1.0 / T;

		if (T <= 0 || std::abs(img) >= 1e-16)
			continue;

		Eigen::Vector3d k_alpha, k_beta, k_gamma;
		k_gamma = (-5.0 * a0 + (-10.0 * v0 - 10.0 * (p0 - pf) * freq) * freq) * freq;
		k_beta = -2.0 * freq * k_gamma;
		k_alpha = -freq * k_beta;

		cost = T * (1.0 + k_gamma.squaredNorm() +
								T * ((k_beta.transpose() * k_gamma)(0) +
										 T * ((k_beta.squaredNorm() + (k_alpha.transpose() * k_gamma)(0)) / 3.0 +
													T * ((k_alpha.transpose() * k_beta)(0) / 4.0 +
															 T * k_alpha.squaredNorm() / 20.0))));
		if (cost <= optimal_cost)
		{
			optimal_cost = cost;
			optimal_T = T;
			coeff.row(3) = k_gamma / 6.0;
			coeff.row(4) = k_beta / 24.0;
			coeff.row(5) = k_alpha / 120.0;
		}
	}

	coeff.row(0) = p0;
	coeff.row(1) = v0;
	coeff.row(2) = a0 / 2.0;

	if (optimal_T == -1.0)
	{
		std::cerr << "StateNode::calcOptimalTrajWithPartialState_: optimal_T cannot be found from point(" << p0(0) << ", "
							<< p0(1) << ", " << p0(2) << ") to point(" << pf(0) << ", " << pf(1) << ", " << pf(2) << ")" << std::endl;
	}

	return optimal_cost;
}

void StateNode::updateStateFromCoeff()
{
	Eigen::Matrix<double, 6, 1> nature_bias;
	nature_bias << 1.0, T_, T_ * T_, T_ * T_ * T_,
			T_ * T_ * T_ * T_, T_ * T_ * T_ * T_ * T_;

	state_.pos = polynomial_coeff.transpose() * nature_bias;

	nature_bias
			<< 0.0,
			1.0, 2.0 * T_, 3.0 * T_ * T_,
			4.0 * T_ * T_ * T_, 5.0 * T_ * T_ * T_ * T_;

	state_.vel = polynomial_coeff.transpose() * nature_bias;

	nature_bias << 0.0, 0.0, 2.0, 6.0 * T_,
			12.0 * T_ * T_, 20.0 * T_ * T_ * T_;

	state_.acc = polynomial_coeff.transpose() * nature_bias;
}

bool StateNode::setParent(Ptr parent, double cost_to_parent, double T, Eigen::Matrix<double, 6, 3> coeff)
{
	parent_ = parent;
	cost_to_parent_ = cost_to_parent;
	cost_ = parent->getCost() + cost_to_parent_;
	T_ = T;
	polynomial_coeff = coeff;

	updateStateFromCoeff();
	return true;
}

bool StateNode::setGoal(DroneState_t goal_state)
{
	double T;
	Eigen::Matrix<double, 6, 3> coeff;
	heuristic_ = calcOptimalTrajWithFullState_(state_, goal_state, T, coeff);
	return true;
}

void KinodynRRTStarPlanner::initPlanner(const PlannerConfig &config, std::shared_ptr<OccupancyMapInterface> map)
{
	goal_tolerance = config.goal_tolerance;
	resolution = config.resolution;
	step_size_ = config.step_size;
	max_iterations_ = config.max_iterations;
	time_limit_sec_ = config.time_limit_sec;
	stop_on_first_feasible_ = config.stop_on_first_feasible;
  safety_margin_ = config.safety_margin;
  safety_weight_ = config.safety_weight;

	map_size_.setZero();
	map_size_(0, 0) = config.map_size(0) / 2.0;
	map_size_(1, 1) = config.map_size(1) / 2.0;
	map_size_(2, 2) = config.map_size(2) / 2.0;
	map_origin_ = config.map_origin;

	std::cout << "------------Kinodynamic RRT* Parameter List------------" << std::endl;
	std::cout << "goal_tolerance:" << goal_tolerance << " m" << std::endl;
	std::cout << "resolution:" << resolution << " s" << std::endl;
	std::cout << "map_size:(" << config.map_size(0) << ", " << config.map_size(1) << ", " << config.map_size(2) << ")" << std::endl;
	std::cout << "map_origin:(" << map_origin_(0) << ", " << map_origin_(1) << ", " << map_origin_(2) << ")" << std::endl;
	std::cout << "step_size:" << step_size_ << " m" << std::endl;
	if (safety_margin_ > 0.0 && safety_weight_ > 0.0) {
		std::cout << "safety_margin:" << safety_margin_ << " m, safety_weight:" << safety_weight_ << std::endl;
	}

	reach_goal_ = false;
	sample_node_ = nullptr;
	total_cost_ = -1.0;
	kd_tree = kd_create(3);
	map_ = map;
	goal_ = new StateNode(Eigen::Vector3d::Zero());
	goal_->setCost(StateNode::kErrorCost);
}

bool KinodynRRTStarPlanner::searchTraj(Eigen::Vector3d start_pos, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc, Eigen::Vector3d end_pos)
{
	// Backward-compatible wrapper: zero terminal velocity/acceleration
	return searchTraj(start_pos, start_vel, start_acc, end_pos, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
}

bool KinodynRRTStarPlanner::searchTraj(Eigen::Vector3d start_pos, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc, Eigen::Vector3d end_pos, Eigen::Vector3d end_vel, Eigen::Vector3d end_acc)
{
	goal_->setState(end_pos, end_vel, end_acc);
	if (!map_ || !map_->hasObservation())
	{
		std::cerr << "KinodynRRTStarPlanner::searchTraj: occupancy map is not ready." << std::endl;
		return false;
	}

	if (start_)
		delete start_;

	start_ = new StateNode(start_pos);
	start_->setState(start_pos, start_vel, start_acc);
	start_->setGoal(goal_->getState());
	updateNode(start_);

	reach_goal_ = false;

	auto t0 = std::chrono::steady_clock::now();
	int iter = 0;

	while (!reach_goal_ && (max_iterations_ <= 0 || iter < max_iterations_))
	{
		if (time_limit_sec_ > 0)
		{
			auto now = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration<double>(now - t0).count();
			if (elapsed >= time_limit_sec_)
			{
				break;
			}
		}

		++iter;

		samplePos();
		double min_cost = StateNode::kErrorCost;
		double optimal_T = -1.0;
		StateNode::Ptr parent = nullptr;
		double T, cost = StateNode::kErrorCost;

		Eigen::Matrix<double, 6, 3> coeff, optimal_coeff;

		nearestBackwardNeighbors();
			for (auto near = neighbors_.cbegin(); near != neighbors_.cend(); ++near)
		{
			cost = sample_node_->calcOptimalTrajWithPartialState((*near)->getState(), T, coeff);
				if (checkCollision(coeff, T))
			{
					double extra = segmentSafetyCost(coeff, T);
					double total = cost + extra;
					if (total < min_cost) {
						min_cost = total;
						parent = (*near);
						optimal_T = T;
						optimal_coeff = coeff;
					}
			}
		}

		if (parent && min_cost < StateNode::kErrorCost)
		{
			sample_node_->setParent(parent, min_cost, optimal_T, optimal_coeff);
			sample_node_->setGoal(goal_->getState());
			updateNode(sample_node_);
		}
		else
		{
			continue;
		}

		nearestForwardNeighbors();
			for (auto near = neighbors_.begin(); near != neighbors_.end(); ++near)
		{
			cost = (*near)->calcOptimalTrajWithFullState(sample_node_->getState(), T, coeff);
				if (checkCollision(coeff, T))
			{
					double extra = segmentSafetyCost(coeff, T);
					double total = cost + extra + sample_node_->getCost();
					if (total < (*near)->getCost()) {
						(*near)->setParent(sample_node_, cost + extra, T, coeff);
					}
			}
		}

			// Connect to goal with full terminal constraints (pos, vel, acc)
			cost = goal_->calcOptimalTrajWithFullState(sample_node_->getState(), T, coeff);
			if (checkCollision(coeff, T))
		{
				double extra = segmentSafetyCost(coeff, T);
				double total = cost + extra + sample_node_->getCost();
				if (total < goal_->getCost()) {
					std::cout << "A feasible trajectory to the goal was found with cost = "
										<< total << std::endl;
					goal_->setParent(sample_node_, cost + extra, T, coeff);
					retrieveTraj(goal_);
					if (stop_on_first_feasible_)
					{
						reach_goal_ = true;
					}
				}
		}
	}

	return !poly_coeff_.empty();
}

double KinodynRRTStarPlanner::calcSampleRadius() const
{
	const int node_count = static_cast<int>(state_nodes_.size()) + 1;
	if (node_count <= 0)
	{
		return step_size_;
	}

	double log_cnt = std::log2(static_cast<double>(node_count)) / static_cast<double>(node_count);
	log_cnt = std::max(log_cnt, 1e-6);
	const double ss_volume = map_size_.determinant() * 8.0;
	const double term = std::max(ss_volume * log_cnt, 1e-6);
	return 3.3812 * std::pow(term, 1.0 / 6.0);
}

void KinodynRRTStarPlanner::samplePos()
{
	int collision_flag = 1;
	Eigen::Vector3d sample_pos;
	Eigen::Vector3d near_pos;
	kdres *nearest_node;

	while (collision_flag == 1)
	{
		sample_pos = Eigen::Vector3d::Random();
		sample_pos = map_size_ * sample_pos + map_origin_;

		double pos[3] = {sample_pos(0), sample_pos(1), sample_pos(2)};
		nearest_node = kd_nearest(kd_tree, pos);
		if (!nearest_node)
		{
			continue;
		}

		auto nearest_ptr = static_cast<StateNode::Ptr>(kd_res_item_data(nearest_node));
		if (!nearest_ptr)
		{
			kd_res_free(nearest_node);
			continue;
		}

		near_pos = nearest_ptr->getState().pos;
		kd_res_free(nearest_node);

		sample_pos = (sample_pos - near_pos).normalized() * step_size_ + near_pos;

		collision_flag = map_->isOccupied(sample_pos) ? 1 : 0;
	}

	sample_node_ = new StateNode(sample_pos);
}

bool KinodynRRTStarPlanner::checkCollision(Eigen::Matrix<double, 6, 3> coeff, double T)
{
	Eigen::Matrix<double, 6, 1> nature_bais;
	Eigen::Vector3d pos, relative_pos;

	for (double t = 0.0; t < T; t += resolution)
	{
		nature_bais << 1.0, t, t * t, t * t * t,
				t * t * t * t, t * t * t * t * t;
		pos = coeff.transpose() * nature_bais;

		relative_pos = pos - map_origin_;
		if (std::abs(relative_pos(0)) > map_size_(0, 0) || std::abs(relative_pos(1)) > map_size_(1, 1) ||
				std::abs(relative_pos(2)) > map_size_(2, 2) || map_->isOccupied(pos))
		{
			return false;
		}
	}
	return true;
}

double KinodynRRTStarPlanner::segmentSafetyCost(Eigen::Matrix<double, 6, 3> coeff, double T) const
{
	if (!(safety_margin_ > 0.0 && safety_weight_ > 0.0) || !map_) return 0.0;
	// Sample along the segment and add soft penalty when distance < safety_margin
	const double dt = std::max(1e-3, resolution);
	Eigen::Matrix<double,6,1> basis;
	double cost = 0.0;
	int n = 0;
	for (double t = 0.0; t <= T; t += dt) {
		basis << 1.0, t, t*t, t*t*t, t*t*t*t, t*t*t*t*t;
		Eigen::Vector3d pos = coeff.transpose() * basis;
		double d = map_->distanceAt(pos);
		if (std::isfinite(d)) {
			double m = safety_margin_ - d;
			if (m > 0.0) {
				cost += m * m; // quadratic penalty inside margin
			}
		}
		++n;
	}
	if (n > 0) cost = cost / static_cast<double>(n); // average
	return safety_weight_ * cost;
}

void KinodynRRTStarPlanner::nearestBackwardNeighbors()
{
	double radius = calcSampleRadius();
	double tau = 0.75 * radius;

	DroneState_t state = sample_node_->getState();
	Eigen::Vector3d p = state.pos, v = state.vel, a = state.acc;

	double state_tau[3] = {0.5 * a(0) * tau * tau - v(0) * tau + p(0),
												 0.5 * a(1) * tau * tau - v(1) * tau + p(1),
												 0.5 * a(2) * tau * tau - v(2) * tau + p(2)};

	kdres *set = kd_nearest_range(kd_tree, state_tau, 4.0 * std::sqrt(5.0) * tau * tau * tau);

	neighbors_.clear();
	StateNode::Ptr neighbor;
	while (!kd_res_end(set))
	{
		neighbor = (StateNode::Ptr)kd_res_item_data(set);
		if (neighbor != sample_node_)
		{
			neighbors_.push_back(neighbor);
		}

		kd_res_next(set);
	}
	kd_res_free(set);
}

void KinodynRRTStarPlanner::nearestForwardNeighbors()
{
	double radius = calcSampleRadius();
	double tau = 0.75 * radius;

	DroneState_t state = sample_node_->getState();
	Eigen::Vector3d p = state.pos, v = state.vel, a = state.acc;

	double state_tau[3] = {0.5 * a(0) * tau * tau + v(0) * tau + p(0),
												 0.5 * a(1) * tau * tau + v(1) * tau + p(1),
												 0.5 * a(2) * tau * tau + v(2) * tau + p(2)};

	kdres *set = kd_nearest_range(kd_tree, state_tau, 4.0 * std::sqrt(5.0) * tau * tau * tau);

	neighbors_.clear();
	StateNode::Ptr neighbor;
	while (!kd_res_end(set))
	{
		neighbor = (StateNode::Ptr)kd_res_item_data(set);

		if (neighbor != sample_node_)
		{
			neighbors_.push_back(neighbor);
		}

		kd_res_next(set);
	}
	kd_res_free(set);
}
