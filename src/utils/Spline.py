import numpy as np
import torch


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def se3_2_qt(wu):
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    B = taylor_B(theta)
    C = taylor_C(theta)
    V = I + B * wx + C * wx @ wx
    t = V @ u[..., None]
    q = exp_r2q(w)
    return q, t


def se3_2_qt_parallel(wu):
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    B = taylor_B(theta)
    C = taylor_C(theta)
    V = I + B * wx + C * wx @ wx
    t = V @ u[..., None]
    q = exp_r2q_parallel(w)
    return q, t.squeeze(-1)


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                      torch.stack([w2, O, -w0], dim=-1),
                      torch.stack([-w1, w0, O], dim=-1)], dim=-2)
    return wx


def taylor_A(x, nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        if i > 0:
            denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_B(x, nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_C(x, nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def exp_r2q(r):
    x, y, z = r[0], r[1], r[2]
    theta = 0.5 * torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    if theta < 1e-20:
        qx = (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * x
        qy = (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * y
        qz = (1. / 2. - 1. / 12. * theta ** 2 - 1. / 240. * theta ** 4) * z
        qw = 1. - 1. / 2. * theta ** 2 + 1. / 24. * theta ** 4
        q_ = torch.stack([qx, qy, qz, qw], 0)
    else:
        lambda_ = torch.sin(theta) / (2. * theta)
        q_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z, torch.cos(theta)], 0)  # xyzw

    return q_


def exp_r2q_parallel(r, eps=1e-9):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    theta = 0.5 * torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    bool_criterion = (theta < eps).unsqueeze(-1).repeat(1, 1, 4)
    return torch.where(
        bool_criterion, exp_r2q_taylor_(x, y, z, theta), exp_r2q_(x, y, z, theta)
    )


def exp_r2q_(x, y, z, theta):
    lambda_ = torch.sin(theta) / (2.0 * theta)
    qx = lambda_ * x
    qy = lambda_ * y
    qz = lambda_ * z
    qw = torch.cos(theta)
    return torch.stack([qx, qy, qz, qw], -1)


def exp_r2q_taylor_(x, y, z, theta):
    qx = (1.0 / 2.0 - 1.0 / 12.0 * theta ** 2 - 1.0 / 240.0 * theta ** 4) * x
    qy = (1.0 / 2.0 - 1.0 / 12.0 * theta ** 2 - 1.0 / 240.0 * theta ** 4) * y
    qz = (1.0 / 2.0 - 1.0 / 12.0 * theta ** 2 - 1.0 / 240.0 * theta ** 4) * z
    qw = 1.0 - 1.0 / 2.0 * theta ** 2 + 1.0 / 24.0 * theta ** 4
    return torch.stack([qx, qy, qz, qw], -1)


def q_to_R(q):
    qb, qc, qd, qa = q.unbind(dim=-1)
    R = torch.stack(
        [torch.stack([1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
         torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)], dim=-1),
         torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)], dim=-1)],
        dim=-2)
    return R


def q_to_R_parallel(q):
    qb, qc, qd, qa = q.unbind(dim=-1)
    R = torch.stack(
        [torch.stack([1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
         torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)], dim=-1),
         torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)], dim=-1)],
        dim=-2)
    return R


def q_to_Q(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    Q_0 = torch.stack([w, -z, y, x], 0).reshape(1, 4)
    Q_1 = torch.stack([z, w, -x, y], 0).reshape(1, 4)
    Q_2 = torch.stack([-y, x, w, z], 0).reshape(1, 4)
    Q_3 = torch.stack([-x, -y, -z, w], 0).reshape(1, 4)
    Q_ = torch.cat([Q_0, Q_1, Q_2, Q_3], 0)

    return Q_


def q_to_Q_parallel(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    Q_0 = torch.stack([w, -z, y, x], -1).unsqueeze(-2)
    Q_1 = torch.stack([z, w, -x, y], -1).unsqueeze(-2)
    Q_2 = torch.stack([-y, x, w, z], -1).unsqueeze(-2)
    Q_3 = torch.stack([-x, -y, -z, w], -1).unsqueeze(-2)
    Q_ = torch.cat([Q_0, Q_1, Q_2, Q_3], -2)

    return Q_


def q_to_q_conj(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    q_conj_ = torch.stack([-x, -y, -z, w], 0)
    return q_conj_


def q_to_q_conj_parallel(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_conj_ = torch.stack([-x, -y, -z, w], -1)
    return q_conj_


def log_q2r(q):  # here fixme
    x, y, z, w = q[0], q[1], q[2], q[3]
    theta = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    if theta < 1e-20:
        lambda_ = 2. / w - 2. / 3. * (theta ** 2) / (w * w * w)
    elif torch.abs(w) < 1e-10:
        if w > 0:
            lambda_ = torch.pi / theta
        else:
            lambda_ = -torch.pi / theta
    else:
        lambda_ = 2. * (torch.arctan(theta / w)) / theta

    r_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z], 0)

    return r_


def log_q2r_parallel(q, eps_theta=1e-20, eps_w=1e-10):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    theta = torch.sqrt(x ** 2 + y ** 2 + z ** 2)

    bool_theta_0 = theta < eps_theta
    bool_w_0 = torch.abs(w) < eps_w
    bool_w_0_left = torch.logical_and(bool_w_0, w < 0)

    lambda_ = torch.where(
        bool_w_0,
        torch.where(
            bool_w_0_left,
            -torch.pi / theta,
            torch.pi / theta
        ),
        torch.where(
            bool_theta_0,
            2.0 / w - 2.0 / 3.0 * (theta ** 2) / (w * w * w),
            2.0 * (torch.arctan(theta / w)) / theta
        ),
    )

    r_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z], -1)

    return r_


def SE3_to_se3(Rt, eps=1e-8):
    R, t = Rt.split([3, 1], dim=-1)
    w = SO3_to_so3(R)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
    u = (invV @ t)[..., 0]
    wu = torch.cat([w, u], dim=-1)
    return wu


def SO3_to_so3(R, eps=1e-7):
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
                ..., None, None] % np.pi
    lnR = 1 / (2 * taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
    w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
    w = torch.stack([w0, w1, w2], dim=-1)
    return w


def SE3_to_se3_N(poses_rt):
    poses_se3_list = []
    for i in range(poses_rt.shape[0]):
        pose_se3 = SE3_to_se3(poses_rt[i])
        poses_se3_list.append(pose_se3)
    poses = torch.stack(poses_se3_list, 0)

    return poses


def se3_to_SE3(wu):
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
    return Rt


def se3_to_SE3_N(poses_wu):
    poses_se3_list = []
    for i in range(poses_wu.shape[0]):
        pose_se3 = se3_to_SE3(poses_wu[i])
        poses_se3_list.append(pose_se3)
    poses = torch.stack(poses_se3_list, 0)

    return poses


def get_pose_interval(num, device):
    # pose time
    pose_time = (torch.arange(num) / (num - 1)).to(device)
    pose_time = torch.clip(pose_time, 0.0001, 0.9999)

    return pose_time.unsqueeze(0)


def noise_T():
    noise = torch.rand(1, 6) * 0.005
    noise_I = se3_to_SE3(noise).squeeze(0)
    bottom = torch.tensor([0., 0., 0., 1.]).reshape(1, 4)
    pose = torch.cat([noise_I, bottom], 0)

    return pose.cuda()


def spline_rotation(q_start, q_end, pose_time):
    q_tau_0 = q_to_Q_parallel(q_to_q_conj_parallel(q_start)) @ q_end[..., None]
    r = pose_time[..., None] * log_q2r_parallel(q_tau_0.squeeze(-1))
    q_t_0 = exp_r2q_parallel(r)
    q_t = q_to_Q_parallel(q_start) @ q_t_0[..., None]

    return q_t.squeeze(-1)


### ESLAM
# spline for eslam
def process_q_t(q_start, q_end, t_start, t_end, num):
    shape0 = q_start.shape[0]
    device = q_start.device
    pose_nums = torch.arange(num).reshape(1, -1).repeat(shape0, 1).to(device)
    seg_pos_x = torch.arange(shape0).reshape([shape0, 1]).repeat(1, num).to(device)

    q_start, q_end = q_start[seg_pos_x], q_end[seg_pos_x]
    trans_start, trans_end = t_start[seg_pos_x], t_end[seg_pos_x]

    pose_time = pose_nums / (num - 1)
    pose_time = torch.clip(pose_time, 0.0001, 0.9999)
    return q_start, q_end, trans_start, trans_end, pose_time


def spline_linear_qt(qt_start, qt_end, NUM):
    q_start, q_end = torch.cat((qt_start[:, 1:4], qt_start[:, :1]), dim=1), torch.cat((qt_end[:, 1:4], qt_end[:, :1]), dim=1)  # need xyzw
    t_start, t_end = qt_start[:, 4:], qt_end[:, 4:]

    q_start, q_end, t_start, t_end, pose_time = process_q_t(q_start, q_end, t_start, t_end, NUM)

    # 1. sample t
    t_t = (1 - pose_time)[..., None] * t_start + pose_time[..., None] * t_end

    # 2. spline q
    q_t = spline_rotation(q_start, q_end, pose_time)  # xyzw

    # 3. xyzw->wxyz
    q_t_wxyz = torch.cat([q_t[:, :, 3:], q_t[:, :, :3]], dim=2)

    qt = torch.cat([q_t_wxyz, t_t], dim=2)

    return qt.reshape(-1, 7)


###
def return_mid(input, spline_num):
    frame_num = input.shape[0] // spline_num
    idx = torch.arange(frame_num) * spline_num + spline_num // 2
    input_mid = input[idx]

    return input_mid


def return_mid_continuous(input, spline_num):
    batch = input.shape[0] // spline_num
    mid_num = spline_num // 2
    start, end = batch * mid_num, batch * (mid_num+1)
    input_mid = input[start:end]

    return input_mid


def return_mid_split(input, spline_num, batch):
    # simple
    total_shape = input.shape[0]
    split_num = total_shape // batch // spline_num  # number of blur images
    mid_num = spline_num // 2

    start_, end_ = batch*mid_num, batch*(mid_num + 1)

    if len(input.shape) > 1:
        input_reshape = input.reshape(split_num, -1, input.shape[-1])
        input_mid = input_reshape[:, start_:end_].reshape(-1, input.shape[-1])
    else:
        input_reshape = input.reshape(split_num, -1)
        input_mid = input_reshape[:, start_:end_].reshape(-1)

    return input_mid


def return_mid_exclude_first(input, spline_num, batch):
    first = input[:batch]
    mid_exclude_first = return_mid_split(input[batch:], spline_num, batch)

    input_mid = torch.cat([first, mid_exclude_first], dim=0)
    return input_mid


def return_blur_color_exclude_first(input, spline_num, batch):
    first = input[:batch]

    blur = input[batch:]
    split_num = blur.shape[0] // batch //spline_num

    results_sharp = blur.reshape(split_num, spline_num, batch, 3)
    results_blur = torch.mean(results_sharp, 1)

    results_blur_final = results_blur.reshape(-1, 3)   ### right

    color = torch.cat([first, results_blur_final], dim=0)

    return color


def return_blur_color(input, spline_num):

    batch_size = input.shape[0] // spline_num
    results_sharp = input.reshape(spline_num, batch_size, -1)
    results_blur = torch.mean(results_sharp, 0)

    return results_blur
