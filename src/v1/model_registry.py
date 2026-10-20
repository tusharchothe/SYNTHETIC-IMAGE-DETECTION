from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class ModelRegistry:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root_dir / "registry.json"
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"models": {}}, indent=2, sort_keys=True), encoding="utf-8")

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            self._ensure_registry()
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save_registry(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def set_active_model(self, model_name: str, model_path: str | Path) -> None:
        source = Path(model_path)
        active_dir = self.root_dir / model_name
        active_dir.mkdir(parents=True, exist_ok=True)
        active_file = active_dir / "production.pt"
        shutil.copy2(str(source), str(active_file))
        payload = self._load_registry()
        payload["models"].setdefault(model_name, {})["active_model"] = str(active_file)
        payload["models"][model_name]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_registry(payload)

    def get_active_path(self, model_name: str) -> Path:
        payload = self._load_registry()
        active = payload.get("models", {}).get(model_name, {}).get("active_model")
        if not active:
            raise FileNotFoundError(f"No active model registered for '{model_name}'.")
        return Path(active)

    def get_backup_path(self, model_name: str) -> Path:
        payload = self._load_registry()
        model_state = payload.get("models", {}).get(model_name, {})
        backup = model_state.get("backup_path")
        if backup:
            backup_path = Path(backup)
            if backup_path.exists():
                return backup_path

        active = self.get_active_path(model_name)
        backup_dir = active.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backups = sorted(list(backup_dir.glob("*.bak")) + list(backup_dir.glob("*.pt")))
        if backups:
            return backups[-1]
        return backup_dir / "latest_backup.bak"

    def promote_candidate(
        self,
        *,
        model_name: str,
        candidate_path: str | Path,
        evaluation_report: dict[str, Any],
        approver: str,
        authorizer: Callable[[str], bool],
        model_version: str | None = None,
    ) -> dict[str, Any]:
        if not authorizer(approver):
            raise PermissionError("Promotion requires an authorized administrator.")
        if not evaluation_report.get("passed"):
            reasons = ", ".join(evaluation_report.get("rejection_reasons", ["unknown evaluation failure"]))
            raise ValueError(f"Candidate rejected for promotion: {reasons}")

        candidate = Path(candidate_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Candidate model not found: {candidate}")

        active_path = self.get_active_path(model_name) if (self.root_dir / model_name / "production.pt").exists() else None
        registry = self._load_registry()
        model_state = registry.setdefault("models", {}).setdefault(model_name, {})

        if active_path is not None:
            backup_dir = self.root_dir / model_name / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_production_backup.bak"
            shutil.copy2(str(active_path), str(backup_path))
            model_state["backup_path"] = str(backup_path)

        target_dir = self.root_dir / model_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "production.pt"
        shutil.copy2(str(candidate), str(target_path))

        version = model_version or f"{model_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        promoted = {
            "model_name": model_name,
            "version": version,
            "approved": True,
            "approver": approver,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_active_model": str(active_path) if active_path is not None else None,
            "current_model_path": str(target_path),
            "evaluation_report": evaluation_report,
        }
        model_state["active_model"] = str(target_path)
        model_state["current_version"] = version
        model_state["history"] = model_state.get("history", []) + [promoted]
        model_state["updated_at"] = promoted["timestamp"]
        self._save_registry(registry)
        return promoted

    def rollback(self, model_name: str, approver: str, authorizer: Callable[[str], bool], *, target_version: str | None = None) -> dict[str, Any]:
        if not authorizer(approver):
            raise PermissionError("Rollback requires an authorized administrator.")
        registry = self._load_registry()
        model_state = registry.setdefault("models", {}).setdefault(model_name, {})
        history = model_state.get("history", [])
        if not history:
            raise ValueError(f"No previous promotion records exist for '{model_name}'.")

        if target_version is None:
            target_record = history[-1]
        else:
            target_record = next((item for item in history if item["version"] == target_version), None)
            if target_record is None:
                raise ValueError(f"Version '{target_version}' was not found in the history for '{model_name}'.")

        backup_path = Path(target_record.get("previous_active_model") or model_state.get("backup_path"))
        if not backup_path.exists():
            raise FileNotFoundError(f"No rollback artifact exists for version '{target_record.get('version')}'.")

        target = self.root_dir / model_name / "production.pt"
        shutil.copy2(str(backup_path), str(target))
        model_state["active_model"] = str(target)
        model_state["last_rollback"] = {
            "restored_version": target_record["version"],
            "restored_by": approver,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save_registry(registry)
        return {"model_name": model_name, "restored_version": target_record["version"], "path": str(target)}


__all__ = ["ModelRegistry"]
