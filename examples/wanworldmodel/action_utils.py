import torch


def action_value_to_tensor(value):
    value = torch.as_tensor(value)
    if value.ndim == 0:
        raise ValueError("Robot action values must include a frame dimension.")
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    elif value.ndim > 2:
        value = value.reshape(value.shape[0], -1)
    return value


def robot_action_to_tensor(robot):
    pieces = []
    for arm in ("left", "right"):
        arm_action = robot.get(arm, {}).get("action", {})
        for key in ("arm_joint", "gripper"):
            value = arm_action.get(key)
            if value is None:
                continue
            pieces.append(action_value_to_tensor(value))
    if len(pieces) == 0:
        return None
    return torch.cat(pieces, dim=-1)
