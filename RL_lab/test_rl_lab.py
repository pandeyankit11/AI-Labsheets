import unittest
import torch
from rl_core import (WarehouseEnv,TwoRobotWarehouseEnv,q_learning_target,q_learning_update,
 dqn_target,epsilon_for_episode,epsilon_for_step,DQNConfig,state_features_multi)
class SingleRobotEnvironmentTests(unittest.TestCase):
 def test_center_door_and_shortest_path(self):
  env=WarehouseEnv();env.reset();path=[env.pos]
  for a in [3,3,3,3]:pos,reward,term,trunc,info=env.step(a);path.append(pos)
  self.assertEqual(path,[(2,0),(2,1),(2,2),(2,3),(2,4)]);self.assertEqual(reward,10.0);self.assertTrue(term);self.assertFalse(trunc)
 def test_boundary_and_obstacle(self):
  env=WarehouseEnv();env.set_position((0,0));p,r,_,_,i=env.step(0);self.assertEqual(p,(0,0));self.assertEqual(r,-1);self.assertTrue(i["invalid_move"])
  env.set_position((1,1));p,r,_,_,i=env.step(3);self.assertEqual(p,(1,1));self.assertEqual(r,-1);self.assertTrue(i["invalid_move"])
 def test_wait_and_truncation(self):
  env=WarehouseEnv(max_steps=1);env.set_position((1,1));p,r,t,tr,i=env.step(4);self.assertEqual(p,(1,1));self.assertEqual(r,-.1);self.assertFalse(t);self.assertTrue(tr);self.assertFalse(i["invalid_move"])
 def test_invalid_action(self):
  with self.assertRaises(ValueError):WarehouseEnv().step(5)
class TDUpdateTests(unittest.TestCase):
 def test_nonterminal_negative_reward(self):
  v,y=q_learning_update(.4,-1,2,.2,.95,False);self.assertAlmostEqual(y,.9);self.assertAlmostEqual(v,.5)
 def test_terminal_masks_bootstrap(self):
  v,y=q_learning_update(.4,5,100,.2,.95,True);self.assertAlmostEqual(y,5);self.assertAlmostEqual(v,1.32)
 def test_truncation_bootstraps(self):self.assertAlmostEqual(q_learning_target(-1,2,.95,False),.9)
 def test_epsilon_endpoints(self):
  self.assertAlmostEqual(epsilon_for_episode(0),1);self.assertAlmostEqual(epsilon_for_episode(1199),.05)
  self.assertAlmostEqual(epsilon_for_step(9600,DQNConfig(12000)),.05)
class TwoRobotEnvironmentTests(unittest.TestCase):
 def test_joint_action_encoding(self):
  for a in range(5):
   for b in range(5):self.assertEqual(TwoRobotWarehouseEnv.decode_joint_action(TwoRobotWarehouseEnv.encode_joint_action(a,b)),(a,b))
  self.assertEqual(state_features_multi(((2,0),(2,4))).shape,(50,))
 def test_same_cell_collision(self):
  e=TwoRobotWarehouseEnv();e.reset((2,1),(2,3));s,r,t,_,i=e.step(e.encode_joint_action(3,2))
  self.assertEqual(s,((2,1),(2,3)));self.assertEqual(r,-10);self.assertTrue(t);self.assertTrue(i["same_cell_collision"]);self.assertFalse(i["swap_collision"])
 def test_swap_collision(self):
  e=TwoRobotWarehouseEnv();e.reset((2,1),(2,2));s,r,t,_,i=e.step(e.encode_joint_action(3,2))
  self.assertEqual(s,((2,1),(2,2)));self.assertTrue(t);self.assertTrue(i["swap_collision"])
 def test_following_into_vacated_cell_allowed(self):
  e=TwoRobotWarehouseEnv();e.reset((2,1),(2,2));s,_,t,_,i=e.step(e.encode_joint_action(3,3))
  self.assertEqual(s,((2,2),(2,3)));self.assertFalse(t);self.assertFalse(i["collision"])
 def test_up_and_wait(self):
  e=TwoRobotWarehouseEnv();e.reset((2,1),(2,2));s,_,t,_,i=e.step(e.encode_joint_action(0,4))
  self.assertEqual(s,((1,1),(2,2)));self.assertFalse(t);self.assertFalse(i["collision"])
 def test_enter_waiting_cell_collision(self):
  e=TwoRobotWarehouseEnv();e.reset((2,1),(2,2));s,_,t,_,i=e.step(e.encode_joint_action(3,4))
  self.assertEqual(s,((2,1),(2,2)));self.assertTrue(t);self.assertTrue(i["same_cell_collision"])
 def test_goal_occupancy_no_repeat_bonus(self):
  e=TwoRobotWarehouseEnv();e.reset((2,4),(1,4));s,_,t,_,i=e.step(e.encode_joint_action(1,0))
  self.assertEqual(s[0],(2,4));self.assertEqual(i["newly_reached_goals"],0);self.assertFalse(t)
 def test_collision_reward_priority(self):
  e=TwoRobotWarehouseEnv();e.reset((2,1),(2,3));_,r,t,_,i=e.step(e.encode_joint_action(3,2));self.assertEqual(r,-10);self.assertTrue(t);self.assertTrue(i["collision"])
class DQNTargetTests(unittest.TestCase):
 def test_nonterminal_truncated_terminal_targets(self):
  r=torch.tensor([-1.,-1.,5.]);q=torch.tensor([2.,2.,100.]);d=torch.tensor([0.,0.,1.])
  self.assertTrue(torch.allclose(dqn_target(r,q,d,.95),torch.tensor([.9,.9,5.])))
 def test_network_output_dimensions(self):
  from rl_core import QNetwork
  self.assertEqual(QNetwork(25,5)(torch.zeros(1,25)).shape,(1,5))
  self.assertEqual(QNetwork(50,25)(torch.zeros(1,50)).shape,(1,25))
if __name__=="__main__":unittest.main(verbosity=2)
