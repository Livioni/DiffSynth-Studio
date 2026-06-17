import torch
import torch.nn.functional as F


def quaternion_to_matrix(quat: torch.Tensor, quat_order: str = "xyzw", eps: float = 1e-8) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {tuple(quat.shape)}.")
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)
    if quat_order == "xyzw":
        x, y, z, w = quat.unbind(dim=-1)
    elif quat_order == "wxyz":
        w, x, y, z = quat.unbind(dim=-1)
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_rotation_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotation_matrix[..., :, 0], rotation_matrix[..., :, 1]], dim=-1)


def quaternion_to_rotation_6d(quat: torch.Tensor, quat_order: str = "xyzw") -> torch.Tensor:
    return matrix_to_rotation_6d(quaternion_to_matrix(quat, quat_order=quat_order))


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def compute_ee6d_delta(current_6d: torch.Tensor, reference_6d: torch.Tensor) -> torch.Tensor:
    r_cur = rotation_6d_to_matrix(current_6d)
    r_ref = rotation_6d_to_matrix(reference_6d)
    return matrix_to_rotation_6d(r_ref.transpose(-1, -2) @ r_cur)


def abs_eef_to_rela(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if action.ndim != 2 or state.ndim != 2 or state.shape[0] != 1:
        raise ValueError(f"Expected action [T,D] and state [1,D], got {tuple(action.shape)} and {tuple(state.shape)}.")
    if action.shape[-1] != 18 or state.shape[-1] != 18:
        raise ValueError("abs_eef_to_rela expects non-gripper eef6d tensors with dim 18.")

    state = state.to(device=action.device, dtype=action.dtype)
    output = torch.zeros_like(action)
    left_pos = [0, 1, 2]
    left_rot = list(range(3, 9))
    right_pos = [9, 10, 11]
    right_rot = list(range(12, 18))

    left_ref = rotation_6d_to_matrix(state[:, left_rot])
    right_ref = rotation_6d_to_matrix(state[:, right_rot])
    output[:, left_pos] = (left_ref.transpose(-1, -2) @ (action[:, left_pos] - state[:, left_pos]).unsqueeze(-1)).squeeze(-1)
    output[:, right_pos] = (right_ref.transpose(-1, -2) @ (action[:, right_pos] - state[:, right_pos]).unsqueeze(-1)).squeeze(-1)
    output[:, left_rot] = compute_ee6d_delta(action[:, left_rot], state[:, left_rot].expand(action.shape[0], -1))
    output[:, right_rot] = compute_ee6d_delta(action[:, right_rot], state[:, right_rot].expand(action.shape[0], -1))
    return output


def robot_state_to_eef6d(robot, quat_order: str = "xyzw") -> torch.Tensor:
    pieces = []
    for arm in ("left", "right"):
        state = robot.get(arm, {}).get("state", {})
        endpose = state.get("endpose")
        gripper = state.get("gripper")
        if endpose is None or gripper is None:
            raise KeyError(f"Missing {arm} endpose or gripper state.")
        endpose = torch.as_tensor(endpose, dtype=torch.float32)
        gripper = torch.as_tensor(gripper, dtype=torch.float32)
        if gripper.ndim == 1:
            gripper = gripper.unsqueeze(-1)
        pieces.extend([endpose[..., :3], quaternion_to_rotation_6d(endpose[..., 3:7], quat_order=quat_order), gripper[..., :1]])
    return torch.cat(pieces, dim=-1)


def relative_eef6d_action_from_state_sequence(eef6d_state: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    if eef6d_state.ndim != 2 or eef6d_state.shape[-1] != 20:
        raise ValueError(f"Expected eef6d_state [T,20], got {tuple(eef6d_state.shape)}.")
    if eef6d_state.shape[0] < horizon:
        raise ValueError(f"Not enough state frames for horizon={horizon}: {eef6d_state.shape[0]}.")
    current_state = eef6d_state[:1]
    future_state = eef6d_state[:horizon]
    current_no_gripper = torch.cat([current_state[:, :9], current_state[:, 10:19]], dim=-1)
    future_no_gripper = torch.cat([future_state[:, :9], future_state[:, 10:19]], dim=-1)
    relative_no_gripper = abs_eef_to_rela(future_no_gripper, current_no_gripper)
    action = torch.cat(
        [
            relative_no_gripper[:, :9],
            future_state[:, 9:10],
            relative_no_gripper[:, 9:18],
            future_state[:, 19:20],
        ],
        dim=-1,
    )
    return current_state, action
