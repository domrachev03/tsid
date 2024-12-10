import tsid
import numpy as np
from pathlib import Path
import pinocchio as pin

tol = 1e-5
filename = str(Path(__file__).resolve().parent)
path = filename + "/../../models/romeo"
urdf = path + "/urdf/romeo.urdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in path)
robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
model = robot.model()
data = robot.data()

srdf = path + "/srdf/romeo_collision.srdf"
pin.loadReferenceConfigurations(model, srdf, False)
q = model.referenceConfigurations["half_sitting"]

q[2] += 0.84
rng = np.random.default_rng()
v = rng.standard_normal(robot.nv)
print("q:", q.transpose())

print("")
print("Test Task Relative Frame task")
print("")
taskFrameEquality = tsid.TaskTwoFramesEquality("frame_eq", robot, "l_wrist", "r_wrist")

Kp = 100 * np.ones(6)
Kd = 20 * np.ones(6)
taskFrameEquality.setKp(Kp)
taskFrameEquality.setKd(Kd)

assert np.linalg.norm(Kp - taskFrameEquality.Kp, 2) < tol
assert np.linalg.norm(Kd - taskFrameEquality.Kd, 2) < tol

t = 0.0
dt = 0.001
max_it = 1000
error_past = 1e100

for i in range(0, max_it):
    robot.computeAllTerms(data, q, v)

    const = taskFrameEquality.compute(t, q, v, data)

    Jpinv = np.linalg.pinv(const.matrix, 1e-5)
    dv = Jpinv.dot(const.vector)

    assert np.linalg.norm(Jpinv.dot(const.matrix), 2) - 1.0 < tol

    v += dt * dv
    q = pin.integrate(model, q, dt * v)
    t += dt

    error = np.linalg.norm(taskFrameEquality.position_error, 2)
    error_past = error
    if error < 1e-8:
        print("Success Convergence")
        break
    if i % 100 == 0:
        print(
            "Time :",
            t,
            "Frame relative position error :",
            error,
            "Frame relative velocity error :",
            np.linalg.norm(taskFrameEquality.velocity_error, 2),
        )

print("")
print("Test Task Relative Frame task with anchor frame")
print("")
taskFrameEquality = tsid.TaskTwoFramesEquality(
    "frame_eq",
    robot,
    "l_wrist",
    "r_wrist",
    pin.XYZQUATToSE3([0.05, 0, 0, 1, 0, 0, 0]),
    pin.XYZQUATToSE3([0, -0.05, 0, 1, 0, 0, 0]),
)

Kp = 100 * np.ones(6)
Kd = 20 * np.ones(6)
taskFrameEquality.setKp(Kp)
taskFrameEquality.setKd(Kd)

assert np.linalg.norm(Kp - taskFrameEquality.Kp, 2) < tol
assert np.linalg.norm(Kd - taskFrameEquality.Kd, 2) < tol

t = 0.0
dt = 0.001
max_it = 1000
error_past = 1e100

for i in range(0, max_it):
    robot.computeAllTerms(data, q, v)

    const = taskFrameEquality.compute(t, q, v, data)

    Jpinv = np.linalg.pinv(const.matrix, 1e-5)
    dv = Jpinv.dot(const.vector)

    assert np.linalg.norm(Jpinv.dot(const.matrix), 2) - 1.0 < tol

    v += dt * dv
    q = pin.integrate(model, q, dt * v)
    t += dt

    error = np.linalg.norm(taskFrameEquality.position_error, 2)
    error_past = error
    if error < 1e-8:
        print("Success Convergence")
        break
    if i % 100 == 0:
        print(
            "Time :",
            t,
            "Frame relative position error :",
            error,
            "Frame relative velocity error :",
            np.linalg.norm(taskFrameEquality.velocity_error, 2),
        )
