"""Gradio UI for the simple no-auth local workflow.

Two tabs
--------
1. Upload & Detect
   - Accept one or more images via a file picker.
   - POST each to /upload and display per-file results in a table.
   - Show a status summary (how many succeeded / failed / duplicates).

2. Pending Review Queue
   - Load the current pending-review queue from /queue.
   - Display image metadata and model predictions in a table.
   - Click a row to preview the image served from /images/{id}/file.

No authentication inputs anywhere.  The API base URL can be changed in the
sidebar so the UI can connect to a remote instance if needed.
"""

from __future__ import annotations

import io
import json
import tempfile
from typing import Any

import httpx
from PIL import Image

from .config import API_BASE_URL, UI_HOST, UI_PORT


# ── HTTP helpers -------------------------------------------------------------

def _upload_files(api_url: str, pil_images: list[Image.Image]) -> dict:
    """POST a list of PIL images to /upload and return the JSON response."""
    files = []
    for idx, img in enumerate(pil_images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files.append(("files", (f"image_{idx}.png", buf.getvalue(), "image/png")))

    response = httpx.post(
        f"{api_url.rstrip('/')}/upload",
        files=files,
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def _fetch_queue(api_url: str, page: int, limit: int) -> dict:
    """GET /queue and return the JSON payload."""
    response = httpx.get(
        f"{api_url.rstrip('/')}/queue",
        params={"page": page, "limit": limit},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _get_image_path(api_url: str, image_id: str) -> str | None:
    """Get the direct file path for an image (avoiding SSRF protection issues)."""
    try:
        response = httpx.get(
            f"{api_url.rstrip('/')}/images/{image_id}/predictions",
            timeout=10.0,
        )
        response.raise_for_status()
        # We can't get the path directly from this endpoint, so we'll use a workaround
        # by fetching the image bytes and saving to a temp file
        return _image_url_as_bytes(api_url, image_id)
    except Exception:
        return None


def _image_url_as_bytes(api_url: str, image_id: str) -> str | None:
    """Fetch image bytes and return as PIL Image (Gradio compatible)."""
    try:
        import tempfile
        response = httpx.get(
            f"{api_url.rstrip('/')}/images/{image_id}/file",
            timeout=30.0,
        )
        response.raise_for_status()
        
        # Save to temp file that Gradio can read
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
            tmp.write(response.content)
            return tmp.name
    except Exception:
        return None


def _submit_review(api_url: str, image_id: str, verified_class: str, confidence: float, 
                    known_generator: str, evidence_source: str, notes: str) -> dict:
    """POST a review to /reviews/submit."""
    params = {
        "image_id": image_id,
        "verified_class": verified_class,
        "confidence": confidence,
    }
    if known_generator and known_generator.strip():
        params["known_generator"] = known_generator.strip()
    if evidence_source and evidence_source.strip():
        params["evidence_source"] = evidence_source.strip()
    if notes and notes.strip():
        params["source_notes"] = notes.strip()
    
    response = httpx.post(
        f"{api_url.rstrip('/')}/reviews/submit",
        params=params,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _build_dataset(api_url: str, version: str, dry_run: bool, require_second_review: bool) -> dict:
    """POST to /datasets/build."""
    response = httpx.post(
        f"{api_url.rstrip('/')}/datasets/build",
        params={
            "version": version,
            "dry_run": dry_run,
            "require_second_review": require_second_review,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def _list_datasets(api_url: str) -> dict:
    """GET /datasets."""
    response = httpx.get(
        f"{api_url.rstrip('/')}/datasets",
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _run_training(api_url: str, dataset_version: str, model_name: str, epochs: int, 
                  batch_size: int, learning_rate: float, dry_run: bool) -> dict:
    """POST to /training/run."""
    response = httpx.post(
        f"{api_url.rstrip('/')}/training/run",
        params={
            "dataset_version": dataset_version,
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dry_run": dry_run,
        },
        timeout=600.0,  # 10 minutes for training
    )
    response.raise_for_status()
    return response.json()


def _list_training_runs(api_url: str) -> dict:
    """GET /training/runs."""
    response = httpx.get(
        f"{api_url.rstrip('/')}/training/runs",
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _get_dashboard_stats(api_url: str) -> dict:
    """Fetch overall statistics for dashboard."""
    try:
        # Get all queue items to compute stats
        response = httpx.get(
            f"{api_url.rstrip('/')}/queue",
            params={"page": 1, "limit": 1000},  # Large limit to get all
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        
        stats = {
            "total_images": len(items),
            "resnet": {"real": 0, "ai": 0, "total": 0, "avg_score": 0.0},
            "clip": {"real": 0, "ai": 0, "total": 0, "avg_score": 0.0},
            "agreement": 0,
            "disagreement": 0,
        }
        
        resnet_scores = []
        clip_scores = []
        
        for item in items:
            preds = item.get("predictions", [])
            resnet_pred = next((p for p in preds if "resnet" in p["model_name"].lower()), None)
            clip_pred = next((p for p in preds if "clip" in p["model_name"].lower()), None)
            
            if resnet_pred:
                pred_class = resnet_pred["predicted_class"]
                if pred_class in ["real", "ai"]:
                    stats["resnet"][pred_class] += 1
                    stats["resnet"]["total"] += 1
                    resnet_scores.append(resnet_pred["raw_score"])
            
            if clip_pred:
                pred_class = clip_pred["predicted_class"]
                if pred_class in ["real", "ai"]:
                    stats["clip"][pred_class] += 1
                    stats["clip"]["total"] += 1
                    clip_scores.append(clip_pred["raw_score"])
            
            # Check agreement
            if resnet_pred and clip_pred:
                if resnet_pred["predicted_class"] == clip_pred["predicted_class"]:
                    stats["agreement"] += 1
                else:
                    stats["disagreement"] += 1
        
        # Calculate averages
        if resnet_scores:
            stats["resnet"]["avg_score"] = sum(resnet_scores) / len(resnet_scores)
        if clip_scores:
            stats["clip"]["avg_score"] = sum(clip_scores) / len(clip_scores)
        
        return stats
    except Exception as exc:
        return {"error": str(exc)}


# ── Tab 1 helpers ------------------------------------------------------------

def _format_upload_results(results: list[dict]) -> tuple[list[list], str]:
    """Convert /upload results into (table_rows, summary_text)."""
    rows = []
    ok = dup = err = 0
    for r in results:
        st = r.get("status", "error")
        if st == "ok":
            ok += 1
        elif st == "duplicate":
            dup += 1
        else:
            err += 1

        preds = r.get("predictions", [])
        resnet = next((p for p in preds if "resnet" in p["model_name"].lower()), None)
        clip   = next((p for p in preds if "clip"   in p["model_name"].lower()), None)

        # Format scores with visual indicators
        def format_prediction(pred):
            if not pred:
                return "", ""
            pred_class = pred["predicted_class"].upper()
            score = pred["raw_score"]
            
            # Add emoji indicators
            if pred_class == "REAL":
                icon = "✅"
            elif pred_class == "AI":
                icon = "🤖"
            else:
                icon = "❓"
            
            # Color-code based on confidence
            if score >= 0.8:
                confidence = "High"
            elif score >= 0.6:
                confidence = "Med"
            else:
                confidence = "Low"
            
            label = f"{icon} {pred_class}"
            score_display = f"{score:.2%} ({confidence})"
            return label, score_display

        resnet_label, resnet_score = format_prediction(resnet)
        clip_label, clip_score = format_prediction(clip)

        # Format status
        status_icon = "✓" if st == "ok" else "⚠" if st == "duplicate" else "✗"
        status_display = f"{status_icon} {st.upper()}"

        rows.append([
            r.get("filename", ""),
            status_display,
            r.get("image_id", r.get("error", "")),
            resnet_label,
            resnet_score,
            clip_label,
            clip_score,
            "; ".join(r.get("warnings", [])) or "—",
        ])

    summary = (
        f"✓ Processed {len(results)} file(s): "
        f"{ok} new, {dup} duplicate(s), {err} error(s)."
    )
    return rows, summary


# ── Tab 2 helpers ------------------------------------------------------------

def _format_queue_rows(items: list[dict]) -> tuple[list[list], list[str]]:
    """Convert queue items into (table_rows, image_id_choices)."""
    rows = []
    ids = []
    for item in items:
        preds = item.get("predictions", [])
        resnet = next((p for p in preds if "resnet" in p["model_name"].lower()), None)
        clip   = next((p for p in preds if "clip"   in p["model_name"].lower()), None)

        # Format predictions with visual indicators
        def format_prediction(pred):
            if not pred:
                return "", ""
            pred_class = pred["predicted_class"].upper()
            score = pred["raw_score"]
            
            # Add emoji indicators
            if pred_class == "REAL":
                icon = "✅"
            elif pred_class == "AI":
                icon = "🤖"
            else:
                icon = "❓"
            
            # Confidence indicator
            if score >= 0.8:
                confidence = "High"
            elif score >= 0.6:
                confidence = "Med"
            else:
                confidence = "Low"
            
            label = f"{icon} {pred_class}"
            score_display = f"{score:.2%} ({confidence})"
            return label, score_display

        resnet_label, resnet_score = format_prediction(resnet)
        clip_label, clip_score = format_prediction(clip)
        
        # Format review status
        review_status = item["review_status"]
        if review_status == "pending":
            status_display = "⏳ Pending"
        elif review_status == "approved":
            status_display = "✓ Approved"
        elif review_status == "rejected":
            status_display = "✗ Rejected"
        else:
            status_display = review_status

        rows.append([
            item["image_id"],
            item["safe_filename"],
            status_display,
            resnet_label,
            resnet_score,
            clip_label,
            clip_score,
            item["uploaded_at"][:19],  # trim microseconds
        ])
        ids.append(item["image_id"])
    return rows, ids


# ── Build app ----------------------------------------------------------------

def build_ui(api_base: str = API_BASE_URL):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install gradio: pip install gradio") from exc

    with gr.Blocks(title="AI Image Detector — Simple Workflow") as demo:
        gr.Markdown(
            "# AI Image Detector — Complete Workflow\n"
            "Upload → Detect → Review → Build Datasets → Train Models"
        )

        api_url = gr.Textbox(
            label="API base URL",
            value=api_base,
            scale=2,
        )

        # ── Tab 1: Upload & Detect -------------------------------------------
        with gr.Tab("Upload & Detect"):
            gr.Markdown(
                "Select one or more images (PNG / JPEG / WebP, max 8 MB each). "
                "Both models run automatically after upload."
            )
            file_input = gr.File(
                label="Images to upload",
                file_count="multiple",
                file_types=["image"],
            )
            upload_btn = gr.Button("Upload & Detect", variant="primary", size="lg")
            upload_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 📊 Detection Results")
            upload_table = gr.Dataframe(
                headers=[
                    "📁 Filename",
                    "📌 Status",
                    "🆔 Image ID / Error",
                    "🔵 ResNet Prediction",
                    "📈 ResNet Confidence",
                    "🟢 CLIP Prediction",
                    "📈 CLIP Confidence",
                    "⚠️ Warnings",
                ],
                interactive=False,
                wrap=True,
                column_widths=["15%", "8%", "20%", "12%", "12%", "12%", "12%", "9%"],
            )

            def do_upload(api: str, files):
                if not files:
                    return "No files selected.", []

                # gr.File in file_count="multiple" gives a list of temp file paths
                pil_images: list[Image.Image] = []
                filenames: list[str] = []
                load_errors: list[list] = []

                file_list = files if isinstance(files, list) else [files]
                for f in file_list:
                    path = f.name if hasattr(f, "name") else str(f)
                    try:
                        img = Image.open(path).convert("RGB")
                        pil_images.append(img)
                        filenames.append(path.split("/")[-1])
                    except Exception as exc:
                        load_errors.append([
                            path.split("/")[-1], "error",
                            f"Cannot open: {exc}", "", "", "", "", "",
                        ])

                if not pil_images:
                    return "All files failed to open.", load_errors

                try:
                    payload = _upload_files(api, pil_images)
                except httpx.ConnectError:
                    return (
                        f"Cannot reach API at {api}. "
                        "Make sure the API server is running.",
                        [],
                    )
                except Exception as exc:
                    return f"Upload failed: {exc}", []

                results = payload.get("results", [])
                # Patch filenames from local paths back to originals
                for i, r in enumerate(results):
                    if i < len(filenames):
                        r.setdefault("filename", filenames[i])

                rows, summary = _format_upload_results(results)
                return summary, load_errors + rows

            upload_btn.click(
                fn=do_upload,
                inputs=[api_url, file_input],
                outputs=[upload_status, upload_table],
            )

        # ── Tab 2: Dashboard -------------------------------------------------
        with gr.Tab("Dashboard"):
            gr.Markdown(
                "### Model Performance Dashboard\n"
                "Compare ResNet-50 and CLIP predictions on pending images."
            )
            
            dashboard_refresh_btn = gr.Button("Refresh Statistics", variant="primary")
            dashboard_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                # ResNet Section
                with gr.Column():
                    gr.Markdown("## 🔵 ResNet-50 Predictions")
                    resnet_total = gr.Number(label="Total Predictions", interactive=False, value=0)
                    resnet_real = gr.Number(label="Predicted Real", interactive=False, value=0)
                    resnet_ai = gr.Number(label="Predicted AI", interactive=False, value=0)
                    resnet_avg_score = gr.Number(label="Average Confidence Score", interactive=False, value=0.0, precision=4)
                    
                    with gr.Row():
                        resnet_real_pct = gr.Textbox(label="Real %", interactive=False, value="0%")
                        resnet_ai_pct = gr.Textbox(label="AI %", interactive=False, value="0%")
                
                # CLIP Section
                with gr.Column():
                    gr.Markdown("## 🟢 CLIP Predictions")
                    clip_total = gr.Number(label="Total Predictions", interactive=False, value=0)
                    clip_real = gr.Number(label="Predicted Real", interactive=False, value=0)
                    clip_ai = gr.Number(label="Predicted AI", interactive=False, value=0)
                    clip_avg_score = gr.Number(label="Average Confidence Score", interactive=False, value=0.0, precision=4)
                    
                    with gr.Row():
                        clip_real_pct = gr.Textbox(label="Real %", interactive=False, value="0%")
                        clip_ai_pct = gr.Textbox(label="AI %", interactive=False, value="0%")
            
            # Agreement Section
            gr.Markdown("---")
            gr.Markdown("## 🔄 Model Agreement Analysis")
            with gr.Row():
                total_images = gr.Number(label="Total Images", interactive=False, value=0)
                agreement_count = gr.Number(label="Models Agree", interactive=False, value=0)
                disagreement_count = gr.Number(label="Models Disagree", interactive=False, value=0)
                agreement_pct = gr.Textbox(label="Agreement Rate", interactive=False, value="0%")
            
            gr.Markdown(
                "**Note:** Agreement means both models predict the same class (real or ai). "
                "Disagreement indicates cases that may need human review."
            )

            def refresh_dashboard(api: str):
                try:
                    stats = _get_dashboard_stats(api)
                    
                    if "error" in stats:
                        return (
                            f"Error: {stats['error']}",
                            0, 0, 0, 0.0, "0%", "0%",  # ResNet
                            0, 0, 0, 0.0, "0%", "0%",  # CLIP
                            0, 0, 0, "0%",  # Agreement
                        )
                    
                    # ResNet stats
                    r_total = stats["resnet"]["total"]
                    r_real = stats["resnet"]["real"]
                    r_ai = stats["resnet"]["ai"]
                    r_avg = stats["resnet"]["avg_score"]
                    r_real_pct = f"{(r_real / r_total * 100):.1f}%" if r_total > 0 else "0%"
                    r_ai_pct = f"{(r_ai / r_total * 100):.1f}%" if r_total > 0 else "0%"
                    
                    # CLIP stats
                    c_total = stats["clip"]["total"]
                    c_real = stats["clip"]["real"]
                    c_ai = stats["clip"]["ai"]
                    c_avg = stats["clip"]["avg_score"]
                    c_real_pct = f"{(c_real / c_total * 100):.1f}%" if c_total > 0 else "0%"
                    c_ai_pct = f"{(c_ai / c_total * 100):.1f}%" if c_total > 0 else "0%"
                    
                    # Agreement stats
                    total = stats["total_images"]
                    agree = stats["agreement"]
                    disagree = stats["disagreement"]
                    agree_pct = f"{(agree / (agree + disagree) * 100):.1f}%" if (agree + disagree) > 0 else "0%"
                    
                    status_msg = (
                        f"✓ Loaded statistics for {total} images.\n"
                        f"ResNet predictions: {r_total} | CLIP predictions: {c_total}\n"
                        f"Models agree on {agree} images, disagree on {disagree} images."
                    )
                    
                    return (
                        status_msg,
                        r_total, r_real, r_ai, r_avg, r_real_pct, r_ai_pct,
                        c_total, c_real, c_ai, c_avg, c_real_pct, c_ai_pct,
                        total, agree, disagree, agree_pct,
                    )
                except httpx.ConnectError:
                    return (
                        f"Cannot reach API at {api}.",
                        0, 0, 0, 0.0, "0%", "0%",
                        0, 0, 0, 0.0, "0%", "0%",
                        0, 0, 0, "0%",
                    )
                except Exception as exc:
                    return (
                        f"Error: {exc}",
                        0, 0, 0, 0.0, "0%", "0%",
                        0, 0, 0, 0.0, "0%", "0%",
                        0, 0, 0, "0%",
                    )

            dashboard_refresh_btn.click(
                fn=refresh_dashboard,
                inputs=[api_url],
                outputs=[
                    dashboard_status,
                    resnet_total, resnet_real, resnet_ai, resnet_avg_score, resnet_real_pct, resnet_ai_pct,
                    clip_total, clip_real, clip_ai, clip_avg_score, clip_real_pct, clip_ai_pct,
                    total_images, agreement_count, disagreement_count, agreement_pct,
                ],
            )

        # ── Tab 3: Pending Review Queue --------------------------------------
        with gr.Tab("Pending Review Queue"):
            gr.Markdown(
                "Images that have been uploaded and processed but not yet reviewed."
            )
            with gr.Row():
                queue_page = gr.Number(value=1, label="Page", precision=0, minimum=1, scale=1)
                queue_limit = gr.Number(value=20, label="Per page", precision=0, minimum=1, maximum=100, scale=1)
                refresh_btn = gr.Button("Refresh queue", variant="primary", scale=2)

            queue_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 📋 Image Queue")
            queue_table = gr.Dataframe(
                headers=[
                    "🆔 Image ID",
                    "📁 Filename",
                    "📌 Review Status",
                    "🔵 ResNet Prediction",
                    "📈 ResNet Confidence",
                    "🟢 CLIP Prediction",
                    "📈 CLIP Confidence",
                    "📅 Uploaded At",
                ],
                interactive=False,
                wrap=True,
                column_widths=["18%", "15%", "10%", "12%", "12%", "12%", "12%", "9%"],
            )

            # State: list of image_ids in current page (for preview lookup)
            queue_ids_state = gr.State([])

            selected_id = gr.Dropdown(
                label="Select image to preview",
                choices=[],
                value=None,
            )
            image_preview = gr.Image(label="Image preview", interactive=False)

            def do_refresh(api: str, page: int, limit: int):
                try:
                    payload = _fetch_queue(api, int(page), int(limit))
                except httpx.ConnectError:
                    return (
                        f"Cannot reach API at {api}.",
                        [],
                        gr.Dropdown(choices=[], value=None),
                        [],
                        None,
                    )
                except Exception as exc:
                    return (
                        f"Error: {exc}",
                        [],
                        gr.Dropdown(choices=[], value=None),
                        [],
                        None,
                    )

                items = payload.get("items", [])
                total = payload.get("total", 0)
                rows, ids = _format_queue_rows(items)
                dropdown = gr.Dropdown(
                    choices=ids,
                    value=ids[0] if ids else None,
                )
                first_preview = _image_url_as_bytes(api, ids[0]) if ids else None
                summary = (
                    f"Showing {len(rows)} of {total} pending images "
                    f"(page {int(page)})."
                )
                return summary, rows, dropdown, ids, first_preview

            def do_preview(api: str, selected: str, ids: list):
                if not selected:
                    return None
                return _image_url_as_bytes(api, selected)

            refresh_btn.click(
                fn=do_refresh,
                inputs=[api_url, queue_page, queue_limit],
                outputs=[queue_status, queue_table, selected_id, queue_ids_state, image_preview],
            )
            selected_id.change(
                fn=do_preview,
                inputs=[api_url, selected_id, queue_ids_state],
                outputs=[image_preview],
            )

        # ── Tab 4: Review & Label --------------------------------------------
        with gr.Tab("Review & Label"):
            gr.Markdown(
                "### Submit Human Reviews\n"
                "Provide human labels for images. Labels are stored separately from model predictions."
            )
            
            with gr.Row():
                review_image_id = gr.Textbox(
                    label="Image ID",
                    placeholder="Paste image ID from queue",
                    scale=2,
                )
                review_load_btn = gr.Button("Load Image", scale=1)
            
            review_image_preview = gr.Image(label="Image to Review", interactive=False)
            
            with gr.Row():
                review_label = gr.Radio(
                    choices=["real", "ai", "uncertain"],
                    label="Human Label",
                    value="real",
                )
                review_confidence = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="Confidence",
                )
            
            with gr.Row():
                review_generator = gr.Textbox(
                    label="Known Generator (optional)",
                    placeholder="e.g., Midjourney, Stable Diffusion",
                )
                review_evidence = gr.Textbox(
                    label="Evidence Source (optional)",
                    placeholder="e.g., Original source URL",
                )
            
            review_notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="Additional context or observations",
                lines=3,
            )
            
            review_submit_btn = gr.Button("Submit Review", variant="primary")
            review_result = gr.Textbox(label="Result", interactive=False)

            def load_review_image(api: str, img_id: str):
                if not img_id or not img_id.strip():
                    return None
                try:
                    return _image_url_as_bytes(api, img_id.strip())
                except Exception:
                    return None

            def submit_review_action(api: str, img_id: str, label: str, conf: float,
                                   gen: str, evid: str, notes: str):
                if not img_id or not img_id.strip():
                    return "Error: Image ID is required"
                
                try:
                    result = _submit_review(
                        api,
                        img_id.strip(),
                        label,
                        conf,
                        gen or "",
                        evid or "",
                        notes or "",
                    )
                    return (
                        f"✓ Review submitted successfully!\n"
                        f"Review ID: {result['review_id']}\n"
                        f"Label: {result['verified_class']}\n"
                        f"Status: {result['review_status']}"
                    )
                except httpx.HTTPStatusError as exc:
                    return f"Error: {exc.response.status_code} - {exc.response.text}"
                except Exception as exc:
                    return f"Error: {exc}"

            review_load_btn.click(
                fn=load_review_image,
                inputs=[api_url, review_image_id],
                outputs=[review_image_preview],
            )
            
            review_submit_btn.click(
                fn=submit_review_action,
                inputs=[
                    api_url, review_image_id, review_label, review_confidence,
                    review_generator, review_evidence, review_notes,
                ],
                outputs=[review_result],
            )

        # ── Tab 5: Dataset Management ----------------------------------------
        with gr.Tab("Dataset Management"):
            gr.Markdown(
                "### Build Versioned Datasets\n"
                "Create training datasets from reviewed images. "
                "Only approved reviews with binary labels (real/ai) are included."
            )
            
            with gr.Row():
                dataset_list_btn = gr.Button("Refresh Dataset List", scale=1)
                dataset_build_btn = gr.Button("Build New Dataset", variant="primary", scale=1)
            
            dataset_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                dataset_version = gr.Textbox(
                    label="Version ID",
                    placeholder="e.g., v0001",
                    value="v0001",
                    scale=2,
                )
                dataset_dry_run = gr.Checkbox(
                    label="Dry run (validate only)",
                    value=False,
                    scale=1,
                )
                dataset_second_review = gr.Checkbox(
                    label="Require 2+ reviews",
                    value=False,
                    scale=1,
                )
            
            dataset_result = gr.JSON(label="Build Result")
            dataset_list = gr.JSON(label="Available Datasets")

            def list_datasets_action(api: str):
                try:
                    result = _list_datasets(api)
                    versions = result.get("versions", [])
                    if not versions:
                        return "No datasets found.", None
                    return f"Found {len(versions)} dataset version(s).", result
                except httpx.ConnectError:
                    return f"Cannot reach API at {api}.", None
                except Exception as exc:
                    return f"Error: {exc}", None

            def build_dataset_action(api: str, version: str, dry: bool, second: bool):
                if not version or not version.strip():
                    return "Error: Version ID is required", None
                
                try:
                    result = _build_dataset(api, version.strip(), dry, second)
                    summary = (
                        f"✓ Dataset {'validated' if dry else 'built'} successfully!\n"
                        f"Version: {result['version']}\n"
                        f"Manifest rows: {result['manifest_count']}\n"
                    )
                    if not dry:
                        summary += f"Location: {result['version_dir']}"
                    return summary, result
                except httpx.HTTPStatusError as exc:
                    return f"Error: {exc.response.status_code} - {exc.response.text}", None
                except Exception as exc:
                    return f"Error: {exc}", None

            dataset_list_btn.click(
                fn=list_datasets_action,
                inputs=[api_url],
                outputs=[dataset_status, dataset_list],
            )
            
            dataset_build_btn.click(
                fn=build_dataset_action,
                inputs=[api_url, dataset_version, dataset_dry_run, dataset_second_review],
                outputs=[dataset_status, dataset_result],
            )

        # ── Tab 6: Training --------------------------------------------------
        with gr.Tab("Training & Models"):
            gr.Markdown(
                "### Train Models\n"
                "Run explicit model training on a dataset version. "
                "Training runs in the current process and may take several minutes."
            )
            
            with gr.Row():
                training_list_btn = gr.Button("Refresh Training Runs", scale=1)
                training_start_btn = gr.Button("Start Training", variant="primary", scale=1)
            
            training_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                training_dataset = gr.Textbox(
                    label="Dataset Version",
                    placeholder="e.g., v0001",
                    value="v0001",
                    scale=2,
                )
                training_model = gr.Dropdown(
                    label="Model",
                    choices=["resnet50", "clip"],
                    value="resnet50",
                    scale=1,
                )
                training_dry_run = gr.Checkbox(
                    label="Dry run",
                    value=False,
                    scale=1,
                )
            
            with gr.Row():
                training_epochs = gr.Number(
                    label="Epochs",
                    value=5,
                    precision=0,
                    minimum=1,
                    maximum=100,
                    scale=1,
                )
                training_batch = gr.Number(
                    label="Batch Size",
                    value=32,
                    precision=0,
                    minimum=1,
                    maximum=256,
                    scale=1,
                )
                training_lr = gr.Number(
                    label="Learning Rate",
                    value=0.0001,
                    precision=6,
                    minimum=0.000001,
                    maximum=1.0,
                    scale=1,
                )
            
            training_result = gr.JSON(label="Training Result")
            training_runs = gr.JSON(label="Recent Training Runs")

            def list_training_action(api: str):
                try:
                    result = _list_training_runs(api)
                    runs = result.get("runs", [])
                    total = result.get("total", 0)
                    if total == 0:
                        return "No training runs found.", None
                    return f"Found {total} training run(s).", result
                except httpx.ConnectError:
                    return f"Cannot reach API at {api}.", None
                except Exception as exc:
                    return f"Error: {exc}", None

            def start_training_action(api: str, dataset: str, model: str, epochs: int,
                                     batch: int, lr: float, dry: bool):
                if not dataset or not dataset.strip():
                    return "Error: Dataset version is required", None
                
                try:
                    result = _run_training(
                        api,
                        dataset.strip(),
                        model,
                        int(epochs),
                        int(batch),
                        float(lr),
                        dry,
                    )
                    
                    if dry:
                        summary = (
                            f"✓ Training validated successfully!\n"
                            f"Model: {result.get('model_name')}\n"
                            f"Dataset: {result.get('dataset_version')}\n"
                            f"Status: {result.get('status')}"
                        )
                    else:
                        summary = (
                            f"✓ Training completed!\n"
                            f"Job ID: {result.get('job_id')}\n"
                            f"Model: {result.get('model_name')}\n"
                            f"Dataset: {result.get('dataset_version')}\n"
                            f"Artifact: {result.get('artifact_path', 'N/A')}"
                        )
                    
                    return summary, result
                except httpx.HTTPStatusError as exc:
                    return f"Error: {exc.response.status_code} - {exc.response.text}", None
                except httpx.ReadTimeout:
                    return (
                        "Training is taking longer than expected. "
                        "Check the training runs list for status.",
                        None,
                    )
                except Exception as exc:
                    return f"Error: {exc}", None

            training_list_btn.click(
                fn=list_training_action,
                inputs=[api_url],
                outputs=[training_status, training_runs],
            )
            
            training_start_btn.click(
                fn=start_training_action,
                inputs=[
                    api_url, training_dataset, training_model, training_epochs,
                    training_batch, training_lr, training_dry_run,
                ],
                outputs=[training_status, training_result],
            )

    return demo


# ── Launch -------------------------------------------------------------------

def launch(
    host: str = UI_HOST,
    port: int = UI_PORT,
    api_base: str = API_BASE_URL,
    share: bool = False,
) -> None:
    demo = build_ui(api_base=api_base)
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
