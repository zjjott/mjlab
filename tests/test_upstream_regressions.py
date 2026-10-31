"""Paired functional checks for the full-checkout optimization experiments."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import Simulation, SimulationCfg

XML = '''<mujoco><worldbody><body pos="0 0 1"><freejoint/>
<geom type="sphere" size=".1" mass="1"/><body pos=".2 0 0">
<joint name="hinge"/><geom type="capsule" size=".03 .1" mass=".2"/>
</body></body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'''
LEVELS = list(RecomputeLevel)[1:]


def make_sim(device='cuda:0'):
    return Simulation(3, SimulationCfg(), model=mujoco.MjModel.from_xml_string(XML), device=device)


def snapshot(sim):
    fields = {}
    for prefix, owner in (('model', sim.wp_model), ('data', sim.wp_data)):
        # Include every direct numeric Warp array, not only expected constants.
        for name, value in vars(owner).items():
            if isinstance(value, wp.array) and value.size:
                try:
                    array = value.numpy().copy()
                    if array.dtype.kind in 'biufc':
                        fields[prefix + '/' + name] = array
                except (TypeError, RuntimeError):
                    pass
    return fields


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('level', LEVELS)
def test_constants_match_eager_with_new_values_and_arrays(level):
    eager, graph = make_sim(), make_sim()
    eager.use_cuda_graph = False
    for sim in (eager, graph):
        sim.expand_model_fields(('body_mass', 'body_ipos', 'dof_armature', 'body_gravcomp'))
    # Exercise first call, cached replay, and replacing arrays after capture.
    for step in range(4):
        if step == 2:
            for sim in (eager, graph):
                sim.expand_model_fields(('body_inertia',))
            assert not graph._constants_graphs
        for sim in (eager, graph):
            sim.model.body_mass[:, 1] *= 1.01
            sim.model.body_ipos[:, 1, 0] += 0.001
            sim.model.dof_armature[:] += 0.001
            sim.model.body_gravcomp[:, 1] = step * 0.1
            sim.data.qpos[:, 2] += 0.01
            sim.data.qvel[:] = 0.03 * step
            qpos, qvel = sim.data.qpos.clone(), sim.data.qvel.clone()
            sim.recompute_constants(level)
            torch.testing.assert_close(sim.data.qpos[:], qpos, rtol=0, atol=0)
            torch.testing.assert_close(sim.data.qvel[:], qvel, rtol=0, atol=0)
        expected, actual = snapshot(eager), snapshot(graph)
        assert len(expected) > 30 and actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-5, atol=2e-6, err_msg=key)
        if step == 3:
            for sim in (eager, graph):
                sim.forward()
                sim.step()
                sim.reset(torch.tensor([1], device=sim.device))
                sim.forward()
            torch.testing.assert_close(eager.data.qpos[:], graph.data.qpos[:], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('level', LEVELS)
def test_constants_cpu_fallback(level):
    sim = make_sim('cpu')
    assert not sim.use_cuda_graph
    before = sim.data.qpos.clone()
    sim.recompute_constants(level)
    assert not sim._constants_graphs
    torch.testing.assert_close(sim.data.qpos[:], before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_bam_optional_fallback_and_noncontiguous(monkeypatch):
    import bam.mjlab as backend
    from types import SimpleNamespace
    # Arbitrary placements, duplicate DOFs, stale invalid IDs, empty worlds.
    types = torch.tensor([[1, 6, 1, 1], [1, 1, 1, 1]], dtype=torch.int32, device='cuda')
    ids = torch.tensor([[2, 9999, 2, 9999], [9999]*4], dtype=torch.int32, device='cuda')
    forces = torch.tensor([[0.5, 100, -0.25, 100], [100]*4], device='cuda')
    counts = torch.tensor([3, 0], dtype=torch.int32, device='cuda')
    receiver = SimpleNamespace(_data=SimpleNamespace(efc=SimpleNamespace(
        type=types, id=ids, force=forces), nefc=counts), _as_tensor=lambda x: x)
    expected = torch.tensor([[0, 0, .25], [0, 0, 0]], device='cuda')
    fn = backend.BamActuator._dof_friction_force
    torch.testing.assert_close(fn(receiver, 3), expected)
    import sys
    monkeypatch.delitem(sys.modules, 'bam._mjlab_friction', raising=False)
    monkeypatch.setitem(sys.modules, 'triton', None)
    torch.testing.assert_close(fn(receiver, 3), expected)
    for name in ('type', 'id', 'force'):
        value = getattr(receiver._data.efc, name)
        # Preserve shape/values but make the row stride noncontiguous.
        larger = torch.zeros((2, 8), dtype=value.dtype, device=value.device)
        larger[:, ::2] = value
        setattr(receiver._data.efc, name, larger[:, ::2])
    torch.testing.assert_close(fn(receiver, 3), expected)
