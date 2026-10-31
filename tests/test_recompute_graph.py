"""Check constants replay after changing model values and rebuilding graphs."""

import mujoco
import mujoco_warp as mjwarp
import pytest
import torch
import warp as wp

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import Simulation, SimulationCfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA graphs")
@pytest.mark.parametrize("level", list(RecomputeLevel)[1:])
def test_constants_replay_and_invalidation(level):
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
    '<geom type="sphere" size="0.1" mass="1"/>'
    "</body></worldbody></mujoco>"
  )
  sim = Simulation(2, SimulationCfg(), model=model, device="cuda:0")
  sim.expand_model_fields(("body_mass", "body_ipos", "dof_armature"))
  sim.recompute_constants(level)
  assert level in sim._constants_graphs
  for scale in (1.1, 0.9):
    sim.model.body_mass[:, 1] *= scale
    sim.model.body_ipos[:, 1, 0] += 0.001
    sim.model.dof_armature[:] += 0.001
    sim.data.qpos[:, 2] += 0.01
    qpos = sim.data.qpos.clone()
    qvel = sim.data.qvel.clone()
    with wp.ScopedDevice(sim.wp_device):
      getattr(mjwarp, level.name)(sim.wp_model, sim.wp_data)
    expected = {
      name: getattr(sim.model, name).clone()
      for name in ("body_subtreemass", "body_invweight0", "dof_invweight0")
    }
    sim.recompute_constants(level)
    for name, value in expected.items():
      torch.testing.assert_close(getattr(sim.model, name)[:], value)
    torch.testing.assert_close(sim.data.qpos[:], qpos, rtol=0, atol=0)
    torch.testing.assert_close(sim.data.qvel[:], qvel, rtol=0, atol=0)
  sim.create_graph()
  assert not sim._constants_graphs
  sim.recompute_constants(level)
  assert level in sim._constants_graphs
