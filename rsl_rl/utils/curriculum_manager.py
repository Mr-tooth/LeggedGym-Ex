from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def linear_schedule(
    target: str,
    start_value: float,
    end_value: float,
    start_iter: Optional[int] = None,
    end_iter: Optional[int] = None,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    mirror_to_cfg: Optional[str] = None,
) -> Dict[str, Any]:
    start_iter, end_iter = _resolve_iteration_window(
        start_iter=start_iter,
        end_iter=end_iter,
        start_s=start_s,
        end_s=end_s,
    )
    return {
        "target": target,
        "start_iter": int(start_iter),
        "end_iter": int(end_iter),
        "start_value": float(start_value),
        "end_value": float(end_value),
        "mode": "linear",
        "mirror_to_cfg": mirror_to_cfg,
    }


def exponential_schedule(
    target: str,
    start_value: float,
    end_value: float,
    start_iter: Optional[int] = None,
    end_iter: Optional[int] = None,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    mirror_to_cfg: Optional[str] = None,
) -> Dict[str, Any]:
    start_iter, end_iter = _resolve_iteration_window(
        start_iter=start_iter,
        end_iter=end_iter,
        start_s=start_s,
        end_s=end_s,
    )
    return {
        "target": target,
        "start_iter": int(start_iter),
        "end_iter": int(end_iter),
        "start_value": float(start_value),
        "end_value": float(end_value),
        "mode": "exponential",
        "mirror_to_cfg": mirror_to_cfg,
    }


class CurriculumManager:
    def __init__(
        self,
        schedules: List[Dict[str, Any]],
        update_interval_iter: int = 1,
        default_root: str = "env",
        contexts: Optional[Dict[str, Any]] = None,
    ):
        self.schedules = schedules
        self.update_interval_iter = max(1, int(update_interval_iter))
        self.default_root = default_root
        self.contexts = contexts or {}
        self._last_update_iter = -1
        self._last_values: Dict[str, float] = {}

    def set_contexts(self, contexts: Dict[str, Any]) -> None:
        self.contexts = contexts

    def update(self, iteration: int, contexts: Optional[Dict[str, Any]] = None) -> None:
        active_contexts = contexts if contexts is not None else self.contexts
        if self.default_root not in active_contexts:
            raise ValueError(
                f"Curriculum default_root '{self.default_root}' is not available in contexts: {list(active_contexts.keys())}"
            )

        current_iter = int(iteration)
        if (
            self._last_update_iter >= 0
            and current_iter - self._last_update_iter < self.update_interval_iter
        ):
            return

        env_noise_changed = False
        for schedule in self.schedules:
            value = self._scheduled_value(schedule, current_iter)
            target = schedule["target"]
            root_key, target_path = self._resolve_target(target, active_contexts)
            root = active_contexts[root_key]
            self._set_by_path(root, target_path, value)
            self._last_values[f"{root_key}.{target_path}"] = float(value)

            mirror_to_cfg = schedule.get("mirror_to_cfg", None)
            if mirror_to_cfg is not None:
                mirror_root_key, mirror_path = self._resolve_target(
                    mirror_to_cfg, active_contexts
                )
                mirror_root = active_contexts[mirror_root_key]
                self._set_by_path(mirror_root, mirror_path, value)

            if root_key == "env":
                if target_path.startswith("reward_scales.") and hasattr(root, "reward_scales"):
                    reward_name = target_path.split(".", 1)[1]
                    self._set_by_path(root, f"cfg.rewards.scales.{reward_name}", value)
                    root.reward_scales[reward_name] = value * root.dt

                if target_path.startswith("cfg.noise.") or target_path.startswith(
                    "noise_scale_vec"
                ):
                    env_noise_changed = True

            if root_key == "alg":
                self._sync_algorithm_runtime(root, target_path, value)

        env = active_contexts[self.default_root]
        if env_noise_changed and hasattr(env, "_get_noise_scale_vec"):
            env.noise_scale_vec = env._get_noise_scale_vec()

        self._last_update_iter = current_iter

    def get_last_values(self) -> Dict[str, float]:
        return dict(self._last_values)

    def _resolve_target(
        self, target: str, contexts: Dict[str, Any]
    ) -> Tuple[str, str]:
        parts = target.split(".", 1)
        if parts[0] in contexts:
            if len(parts) == 1:
                raise ValueError(f"Invalid curriculum target path: '{target}'")
            return parts[0], parts[1]
        return self.default_root, target

    def _sync_algorithm_runtime(self, alg: Any, target_path: str, value: Any) -> None:
        if target_path == "learning_rate":
            self._set_optimizer_lr(alg, "optimizer", float(value))
            return

        if target_path == "encoder_lr":
            self._set_optimizer_lr(alg, "history_encoder_optimizer", float(value))
            self._set_optimizer_lr(alg, "vae_optimizer", float(value))
            return

        if target_path == "estimator_lr":
            self._set_optimizer_lr(alg, "estimator_optimizer", float(value))
            return

        if target_path.startswith("optimizer.param_groups") and target_path.endswith(".lr"):
            if hasattr(alg, "optimizer") and len(alg.optimizer.param_groups) > 0:
                alg.learning_rate = float(alg.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _set_optimizer_lr(alg: Any, optimizer_attr: str, lr: float) -> None:
        if not hasattr(alg, optimizer_attr):
            return
        optimizer = getattr(alg, optimizer_attr)
        if optimizer is None:
            return
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _scheduled_value(self, schedule: Dict[str, Any], iteration: int) -> float:
        start_iter = int(schedule["start_iter"])
        end_iter = int(schedule["end_iter"])
        start_value = float(schedule["start_value"])
        end_value = float(schedule["end_value"])
        mode = schedule.get("mode", "linear")

        if end_iter <= start_iter:
            return end_value
        ratio = (float(iteration) - float(start_iter)) / float(end_iter - start_iter)
        ratio = max(0.0, min(1.0, ratio))

        if mode == "exponential":
            if start_value > 0.0 and end_value > 0.0:
                return start_value * math.pow(end_value / start_value, ratio)
            return start_value + (end_value - start_value) * ratio

        return start_value + (end_value - start_value) * ratio

    def _set_by_path(self, root: Any, path: str, value: Any) -> None:
        if path.startswith("env."):
            path = path[4:]

        parts = path.split(".")
        cursor = root
        for part in parts[:-1]:
            cursor = self._descend(cursor, part)

        last = parts[-1]
        if isinstance(cursor, dict):
            cursor[last] = value
            return

        if self._is_int(last) and isinstance(cursor, list):
            cursor[int(last)] = value
            return

        setattr(cursor, last, value)

    @staticmethod
    def _descend(cursor: Any, key: str) -> Any:
        if isinstance(cursor, dict):
            return cursor[key]

        if CurriculumManager._is_int(key) and isinstance(cursor, list):
            return cursor[int(key)]

        return getattr(cursor, key)

    @staticmethod
    def _is_int(text: str) -> bool:
        try:
            int(text)
        except ValueError:
            return False
        return True


def _resolve_iteration_window(
    start_iter: Optional[int],
    end_iter: Optional[int],
    start_s: Optional[float],
    end_s: Optional[float],
) -> tuple[int, int]:
    if start_iter is None:
        start_iter = 0 if start_s is None else int(round(float(start_s)))
    if end_iter is None:
        end_iter = int(start_iter) if end_s is None else int(round(float(end_s)))
    return int(start_iter), int(end_iter)
