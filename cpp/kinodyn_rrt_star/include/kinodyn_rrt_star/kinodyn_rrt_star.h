#pragma once
#include <algorithm>
#include <limits>
#include <list>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "kdtree/kdtree.h"

typedef struct DroneState
{
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
  Eigen::Vector3d acc;
} DroneState_t;

class OccupancyMapInterface
{
public:
  virtual ~OccupancyMapInterface() = default;
  virtual bool hasObservation() const = 0;
  virtual bool isOccupied(const Eigen::Vector3d &position) const = 0;
  // Optional: signed distance (meters). Return +inf if not available.
  virtual double distanceAt(const Eigen::Vector3d &position) const { return std::numeric_limits<double>::infinity(); }
};

struct PlannerConfig
{
  double goal_tolerance = 0.1;
  double resolution = 0.05;
  Eigen::Vector3d map_size = Eigen::Vector3d(10.0, 10.0, 10.0);
  Eigen::Vector3d map_origin = Eigen::Vector3d::Zero();
  double step_size = 1.0;
  // 新增：搜索停止条件
  int max_iterations = 2000;         // 迭代上限
  double time_limit_sec = 2.0;       // 时间上限（秒），<=0 表示不启用
  bool stop_on_first_feasible = true;// 找到第一条可行解即停止
  // ESDF 安全代价
  double safety_margin = 0.0;        // 米；>0 时启用软安全代价
  double safety_weight = 0.0;        // 权重；越大越避障
};

class StateNode
{
public:
  static constexpr double kErrorCost = std::numeric_limits<double>::max();

  StateNode(Eigen::Vector3d pos) : cost_(0.0), heuristic_(0.0), parent_(nullptr), children_({nullptr})
  {
    state_.pos = pos;
    state_.vel = Eigen::Vector3d(0.0, 0.0, 0.0);
    state_.acc = Eigen::Vector3d(0.0, 0.0, 0.0);
  };
  typedef StateNode *Ptr;

  double calcOptimalTrajWithFullState(DroneState_t parent, double &optimal_T, Eigen::Matrix<double, 6, 3> &coeff) { return calcOptimalTrajWithFullState_(parent, state_, optimal_T, coeff); };
  double calcOptimalTrajWithPartialState(DroneState_t parent, double &optimal_T, Eigen::Matrix<double, 6, 3> &coeff) { return calcOptimalTrajWithPartialState_(parent, state_.pos, optimal_T, coeff); };

  void setCostToParent(double cost) { cost_to_parent_ = cost; };
  void setCost(double cost) { cost_ = cost; };

  bool setParent(Ptr parent, double cost_to_parent, double T, Eigen::Matrix<double, 6, 3> coeff);
  bool setGoal(DroneState_t goal_state);
  void setState(Eigen::Vector3d pos, Eigen::Vector3d vel, Eigen::Vector3d acc)
  {
    state_.pos = pos;
    state_.vel = vel;
    state_.acc = acc;
  };

  Ptr getParent() { return parent_; };
  DroneState_t getState() { return state_; };
  double getCostToParent() { return cost_to_parent_; };
  double getCost() { return cost_; };
  double getHeuristic() { return heuristic_; };
  Eigen::Matrix<double, 6, 3> getPolyTraj() { return polynomial_coeff; };
  double getInterval() { return T_; };

  bool operator<(StateNode node)
  {
    return cost_ + heuristic_ < node.getCost() + node.getHeuristic();
  };

private:
  DroneState_t state_;
  double cost_to_parent_;
  double cost_;
  double heuristic_;
  Eigen::Matrix<double, 6, 3> polynomial_coeff;
  double T_;
  Ptr parent_;
  std::vector<Ptr> children_;

  double calcOptimalTrajWithFullState_(DroneState_t start, DroneState_t end, double &optimal_T, Eigen::Matrix<double, 6, 3> &coeff);
  double calcOptimalTrajWithPartialState_(DroneState_t start, Eigen::Vector3d pf, double &optimal_T, Eigen::Matrix<double, 6, 3> &coeff);
  void updateStateFromCoeff();
};

class KinodynRRTStarPlanner
{
public:
  KinodynRRTStarPlanner(){};
  void initPlanner(const PlannerConfig &config, std::shared_ptr<OccupancyMapInterface> map);
  // Search with goal position only (backward-compatible): goal velocity/acceleration assumed zero
  bool searchTraj(Eigen::Vector3d start_pos, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                  Eigen::Vector3d end_pos);
  // Search with full terminal constraints (preferred): specify desired goal velocity and acceleration
  bool searchTraj(Eigen::Vector3d start_pos, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                  Eigen::Vector3d end_pos, Eigen::Vector3d end_vel, Eigen::Vector3d end_acc);
  std::vector<Eigen::Matrix<double, 6, 3>> getTrajCoeff() const { return poly_coeff_; };
  std::vector<double> getTrajInterval() const { return intervals_; };
  double getCost() const { return total_cost_; };
  void setMap(std::shared_ptr<OccupancyMapInterface> map) { map_ = map; };

private:
  bool reach_goal_;
  double goal_tolerance;
  double resolution;
  double step_size_;
  int max_iterations_;
  double time_limit_sec_;
  bool stop_on_first_feasible_;
  double safety_margin_;
  double safety_weight_;
  StateNode::Ptr start_;
  StateNode::Ptr goal_;
  StateNode::Ptr sample_node_;
  std::vector<StateNode::Ptr> neighbors_;
  std::list<StateNode::Ptr> state_nodes_;

  std::vector<Eigen::Matrix<double, 6, 3>> poly_coeff_;
  std::vector<double> intervals_;
  double total_cost_;

  Eigen::Matrix3d map_size_;
  Eigen::Vector3d map_origin_;
  std::shared_ptr<OccupancyMapInterface> map_;

  kdtree *kd_tree;

  void samplePos();
  double calcSampleRadius() const;
  void nearestBackwardNeighbors();
  void nearestForwardNeighbors();
  bool checkCollision(Eigen::Matrix<double, 6, 3> coeff, double T);
  double segmentSafetyCost(Eigen::Matrix<double, 6, 3> coeff, double T) const;

  bool checkGoal(Eigen::Vector3d pos, Eigen::Vector3d goal)
  {
    double distance = (pos - goal).norm();
    return (distance <= goal_tolerance);
  };

  StateNode::Ptr getCloestNode()
  {
    state_nodes_.sort();
    return (state_nodes_.front());
  };

  void retrieveTraj(StateNode::Ptr node)
  {
    total_cost_ = node->getCost();
    while (node->getParent() != nullptr)
    {
      poly_coeff_.push_back(node->getPolyTraj());
      intervals_.push_back(node->getInterval());
      node = node->getParent();
    }
    reverse(poly_coeff_.begin(), poly_coeff_.end());
    reverse(intervals_.begin(), intervals_.end());
  };

  void updateNode(StateNode::Ptr node)
  {
    state_nodes_.push_back(node);
    DroneState_t state = node->getState();
    double *pt = new double[3]{state.pos(0), state.pos(1), state.pos(2)};
    kd_insert(kd_tree, pt, node);
  };

  void reBuildKdTree()
  {
    kd_clear(kd_tree);
    DroneState_t state;
    for (auto node = state_nodes_.cbegin(); node != state_nodes_.cend(); ++node)
    {
      state = (*node)->getState();
      double *pt = new double[3]{state.pos(0), state.pos(1), state.pos(2)};
      kd_insert(kd_tree, pt, *node);
    }
  };
};
