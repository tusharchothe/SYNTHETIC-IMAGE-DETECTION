from __future__ import annotations

import asyncio
import io
import os
from typing import Tuple

import httpx
from PIL import Image

# Load .env values so default field values (reviewer ID, token, API URL) are
# populated automatically when the Gradio app starts.
try:
    from .config import _load_default_env
    _load_default_env()
except ImportError:
    try:
        from v1.config import _load_default_env
        _load_default_env()
    except ImportError:
        pass


def _model_output(prob: float, label: str) -> Tuple[dict, str]:
    scores = {"real": round(1.0 - prob, 4), "ai": round(prob, 4)}
    message = f"Prediction: {label.upper()} | confidence={round(prob, 4)}"
    return scores, message


def predict_with_model(image: Image.Image | None, model_type: str) -> Tuple[dict, str]:
    if image is None:
        return {"real": 0.0, "ai": 0.0}, "Please upload an image first."

    model_type = (model_type or "resnet").lower()
    if model_type not in {"resnet", "clip"}:
        return {"real": 0.0, "ai": 0.0}, "Model type must be 'resnet' or 'clip'."

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    try:
        from .services.inference_service import run_prediction_for_model

        inference = run_prediction_for_model(model_type, payload)
        return _model_output(inference.raw_score, inference.predicted_class)
    except Exception as exc:  # pragma: no cover
        return {"real": 0.0, "ai": 0.0}, f"Missing ML dependencies or inference error: {exc}"


def _api_headers(reviewer_id: str, token: str) -> dict:
    headers = {"X-Reviewer-Id": reviewer_id}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def save_image_to_review_queue(api_base_url: str, reviewer_id: str, token: str, image: Image.Image, source: str = "manual") -> str:
    if image is None:
        return "Please upload an image before saving it."

    if not api_base_url:
        api_base_url = "http://localhost:8000"

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    response = httpx.post(
        f"{api_base_url.rstrip('/')}/images/upload",
        files={"file": ("uploaded_image.png", payload, "image/png")},
        data={"source": source or "manual"},
        headers=_api_headers(reviewer_id, token),
        timeout=60.0,
    )
    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else "Upload failed."
        return f"Failed to save image to review queue: {detail}"

    body = response.json()
    image_id = body.get("image_id", "unknown")
    return f"Image saved to review queue: {image_id}"


def fetch_review_queue(api_base_url: str, page: int, status: str, reviewer_id: str, token: str):
    if not api_base_url:
        api_base_url = "http://localhost:8000"
    response = httpx.get(
        f"{api_base_url.rstrip('/')}/reviews/queue",
        params={"page": page, "limit": 10, "status": status},
        headers=_api_headers(reviewer_id, token),
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text) if response.content else "Unable to load review queue.")

    payload = response.json()
    base = api_base_url.rstrip("/")
    rows = []
    for item in payload.get("items", []):
        resnet = item.get("resnet") or {}
        clip = item.get("clip") or {}
        # Attach a served URL so Gradio can fetch the image over HTTP instead of
        # relying on a local filesystem path that may not exist on the client.
        item["_preview_url"] = f"{base}/images/{item['image_id']}/file"
        rows.append(
            [
                item.get("image_id", ""),
                item.get("review_status", ""),
                resnet.get("predicted_class", ""),
                clip.get("predicted_class", ""),
                str(resnet.get("raw_score", "")),
                str(clip.get("raw_score", "")),
                item.get("uploaded_at", ""),
            ]
        )
    return rows, payload.get("items", []), payload.get("total", 0)


def submit_review(api_base_url: str, image_id: str, reviewer_id: str, token: str, decision: str, generator: str, confidence: float, evidence_source: str, notes: str):
    if not image_id:
        return "Select an image before submitting a review."
    response = httpx.post(
        f"{api_base_url.rstrip('/')}/reviews/submit",
        json={
            "image_id": image_id,
            "decision": decision,
            "known_generator": generator if generator and generator != "Unknown" else None,
            "confidence": confidence,
            "evidence_source": evidence_source or None,
            "source_notes": notes or None,
        },
        headers=_api_headers(reviewer_id, token),
        timeout=30.0,
    )
    if response.status_code == 200:
        return response.json().get("message", "Review submitted successfully.")
    detail = response.json().get("detail", response.text) if response.content else "Review submission failed."
    return f"Review submission failed: {detail}"


def fetch_dashboard_overview(api_base_url: str, reviewer_id: str, token: str, start_date: str, end_date: str, model_version: str, generator: str, source: str, label: str, review_status: str):
    if not api_base_url:
        api_base_url = "http://localhost:8000"
    response = httpx.get(
        f"{api_base_url.rstrip('/')}/dashboard/overview",
        params={
            "start_date": start_date or None,
            "end_date": end_date or None,
            "model_version": model_version or None,
            "generator": generator or None,
            "source": source or None,
            "label": label or None,
            "review_status": review_status or None,
        },
        headers=_api_headers(reviewer_id, token),
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text) if response.content else "Unable to load dashboard overview.")
    return response.json()


def export_dashboard_csv(api_base_url: str, reviewer_id: str, token: str):
    if not api_base_url:
        api_base_url = "http://localhost:8000"
    response = httpx.get(
        f"{api_base_url.rstrip('/')}/dashboard/export",
        params={"format": "csv"},
        headers=_api_headers(reviewer_id, token),
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text) if response.content else "Unable to export dashboard CSV.")
    return response.content


def fetch_model_lifecycle(api_base_url: str, reviewer_id: str, token: str):
    if not api_base_url:
        api_base_url = "http://localhost:8000"
    response = httpx.get(
        f"{api_base_url.rstrip('/')}/dashboard/model-lifecycle",
        headers=_api_headers(reviewer_id, token),
        timeout=30.0,
    )
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text) if response.content else "Unable to load model lifecycle.")
    return response.json()


def build_gradio_app():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install gradio to use the UI: pip install gradio") from exc

    with gr.Blocks(title="AI Image Detector") as demo:
        gr.Markdown("# AI Image Detector\nUpload an image and choose either the ResNet or CLIP detector.")

        reviewer_id = gr.Textbox(label="Reviewer ID", value=os.getenv("AIDETECT_REVIEWER_ID", "reviewer-1"))
        review_token = gr.Textbox(label="Review token", type="password", value=os.getenv("AIDETECT_REVIEWER_TOKEN", ""))
        api_base_url = gr.Textbox(label="API base URL", value=os.getenv("AIDETECT_API_BASE_URL", "http://localhost:8000"))

        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Input image")
                    model_choice = gr.Dropdown(["resnet", "clip"], value="resnet", label="Model")
                    submit_btn = gr.Button("Detect")
                with gr.Column():
                    output_label = gr.Label(label="Prediction probabilities")
                    output_text = gr.Textbox(label="Result")
                    save_review_btn = gr.Button("Save to review queue")
                    save_status = gr.Textbox(label="Save status")

            submit_btn.click(fn=predict_with_model, inputs=[image_input, model_choice], outputs=[output_label, output_text])

            def save_prediction_to_queue(api_url: str, reviewer_id_value: str, token_value: str, image_value: Image.Image):
                try:
                    return save_image_to_review_queue(api_url, reviewer_id_value, token_value, image_value, source="manual")
                except Exception as exc:  # pragma: no cover
                    return f"Failed to save image to review queue: {exc}"

            save_review_btn.click(
                fn=save_prediction_to_queue,
                inputs=[api_base_url, reviewer_id, review_token, image_input],
                outputs=[save_status],
            )

        with gr.Tab("Review Dashboard"):
            review_filter = gr.Dropdown(
                [
                    "pending",
                    "reviewed",
                    "model_disagreement",
                    "low_confidence",
                    "suspected_false_positive",
                    "suspected_false_negative",
                ],
                value="pending",
                label="Review filter",
            )
            page_number = gr.Number(value=1, precision=0, minimum=1, label="Page")
            refresh_queue_btn = gr.Button("Load queue")
            queue_state = gr.State([])
            queue_table = gr.Dataframe(
                headers=["image_id", "status", "resnet", "clip", "resnet_score", "clip_score", "uploaded_at"],
                interactive=False,
            )
            selected_image = gr.Dropdown(label="Selected image ID", choices=[], value=None)
            image_preview = gr.Image(label="Image preview")

            with gr.Row():
                decision = gr.Radio(["real", "ai", "uncertain"], value="real", label="Decision")
                generator = gr.Dropdown(
                    ["Unknown", "sd14", "sd15", "wukong", "adm", "glide", "vqdm", "midjourney", "biggan"],
                    value="Unknown",
                    label="Generator (optional)",
                )
                confidence = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Confidence")
            evidence_source = gr.Textbox(label="Evidence/source notes")
            notes = gr.Textbox(lines=4, label="Source notes")
            submit_review_btn = gr.Button("Submit review")
            review_message = gr.Markdown()

            def load_queue(api_url: str, page: int, status: str, reviewer_id_value: str, token_value: str):
                try:
                    import gradio as gr
                    rows, items, total = fetch_review_queue(api_url, page, status, reviewer_id_value, token_value)
                    selected_choices = [item["image_id"] for item in items]
                    # Use the served HTTP URL for the initial preview, not the local path
                    preview_value = items[0].get("_preview_url") or items[0].get("storage_path") if items else None
                    # Return a gr.Dropdown update so choices AND value are set atomically,
                    # preventing the "value not in choices" stale-state error.
                    dropdown_update = gr.Dropdown(
                        choices=selected_choices,
                        value=selected_choices[0] if selected_choices else None,
                    )
                    return rows, items, dropdown_update, preview_value, f"Loaded {total} review items."
                except Exception as exc:
                    import gradio as gr
                    return [], [], gr.Dropdown(choices=[], value=None), None, f"Error loading review queue: {exc}"

            def load_image_preview(selected_id: str, items):
                if not selected_id or not items:
                    return None
                for item in items:
                    if item.get("image_id") == selected_id:
                        # Prefer the served HTTP URL; fall back to local path only
                        # when both Gradio and the API are on the same machine.
                        return item.get("_preview_url") or item.get("storage_path")
                return None

            refresh_queue_btn.click(
                fn=load_queue,
                inputs=[api_base_url, page_number, review_filter, reviewer_id, review_token],
                outputs=[queue_table, queue_state, selected_image, image_preview, review_message],
            )
            selected_image.change(fn=load_image_preview, inputs=[selected_image, queue_state], outputs=[image_preview])

            def submit_review_action(api_url: str, image_id: str, reviewer_id_value: str, token_value: str, decision_value: str, generator_value: str, confidence_value: float, evidence_source_value: str, notes_value: str):
                try:
                    return submit_review(api_url, image_id, reviewer_id_value, token_value, decision_value, generator_value, confidence_value, evidence_source_value, notes_value)
                except Exception as exc:  # pragma: no cover
                    return f"Review submission failed: {exc}"

            submit_review_btn.click(
                fn=submit_review_action,
                inputs=[api_base_url, selected_image, reviewer_id, review_token, decision, generator, confidence, evidence_source, notes],
                outputs=[review_message],
            )

        with gr.Tab("Dashboard Overview"):
            dashboard_start = gr.Textbox(label="Start date (ISO)", placeholder="2026-01-01T00:00:00")
            dashboard_end = gr.Textbox(label="End date (ISO)", placeholder="2026-12-31T23:59:59")
            dashboard_model_version = gr.Textbox(label="Model version")
            dashboard_generator = gr.Textbox(label="Generator")
            dashboard_source = gr.Textbox(label="Source")
            dashboard_label = gr.Dropdown(["real", "ai"], label="Label")
            dashboard_status = gr.Dropdown(["pending", "needs_review", "approved", "rejected", "duplicate"], label="Review status")
            with gr.Row():
                load_dashboard_btn = gr.Button("Load dashboard")
                export_csv_btn = gr.Button("Export CSV")
            dashboard_json = gr.JSON(label="Dashboard summary")
            dashboard_csv = gr.File(label="Exported CSV")
            dashboard_warning = gr.Markdown()

            # Load overview only — does NOT trigger the CSV export
            def load_dashboard(api_url: str, reviewer_id_value: str, token_value: str, start_date: str, end_date: str, model_version: str, generator: str, source: str, label: str, review_status: str):
                try:
                    payload = fetch_dashboard_overview(api_url, reviewer_id_value, token_value, start_date, end_date, model_version, generator, source, label, review_status)
                    warnings = "\n".join(f"- {warning}" for warning in payload.get("warnings", [])) or "No warnings."
                    return payload, warnings
                except Exception as exc:  # pragma: no cover
                    return {"error": str(exc)}, f"Error loading dashboard: {exc}"

            load_dashboard_btn.click(
                fn=load_dashboard,
                inputs=[api_base_url, reviewer_id, review_token, dashboard_start, dashboard_end, dashboard_model_version, dashboard_generator, dashboard_source, dashboard_label, dashboard_status],
                outputs=[dashboard_json, dashboard_warning],
            )

            # Export CSV only — independent from the overview load
            def do_export_csv(api_url: str, reviewer_id_value: str, token_value: str):
                try:
                    import tempfile, os
                    csv_bytes = export_dashboard_csv(api_url, reviewer_id_value, token_value)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
                    tmp.write(csv_bytes)
                    tmp.close()
                    return tmp.name
                except Exception as exc:  # pragma: no cover
                    return None

            export_csv_btn.click(
                fn=do_export_csv,
                inputs=[api_base_url, reviewer_id, review_token],
                outputs=[dashboard_csv],
            )

        with gr.Tab("Model Lifecycle"):
            lifecycle_api_url = gr.Textbox(label="API base URL", value=os.getenv("AIDETECT_API_BASE_URL", "http://localhost:8000"))
            lifecycle_reviewer = gr.Textbox(label="Reviewer ID", value=os.getenv("AIDETECT_REVIEWER_ID", "reviewer-1"))
            lifecycle_token = gr.Textbox(label="Review token", type="password", value=os.getenv("AIDETECT_REVIEWER_TOKEN", ""))
            lifecycle_btn = gr.Button("Load model lifecycle")
            lifecycle_json = gr.JSON(label="Model lifecycle")
            lifecycle_warning = gr.Markdown()

            def load_lifecycle(api_url: str, reviewer_id_value: str, token_value: str):
                try:
                    payload = fetch_model_lifecycle(api_url, reviewer_id_value, token_value)
                    return payload, ""
                except Exception as exc:  # pragma: no cover
                    return {"error": str(exc)}, f"Error loading lifecycle: {exc}"

            lifecycle_btn.click(
                fn=load_lifecycle,
                inputs=[lifecycle_api_url, lifecycle_reviewer, lifecycle_token],
                outputs=[lifecycle_json, lifecycle_warning],
            )

    return demo


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    demo = build_gradio_app()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
