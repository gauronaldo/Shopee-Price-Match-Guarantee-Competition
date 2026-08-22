"""Streamlit client for guided and user-supplied product-matching queries."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

API_URL = os.getenv("SHOPEE_DEMO_API_URL", "http://localhost:8000").rstrip("/")
PUBLIC_API_URL = os.getenv("SHOPEE_DEMO_PUBLIC_API_URL", API_URL).rstrip("/")


def _health() -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _guided_samples() -> list[dict[str, str]]:
    response = httpx.get(f"{API_URL}/api/v1/guided-samples", timeout=10.0)
    response.raise_for_status()
    return response.json()


def _catalog_image_url(posting_id: str, *, public: bool) -> str:
    base = PUBLIC_API_URL if public else API_URL
    return f"{base}/api/v1/catalog/{quote(posting_id, safe='')}/image"


def _catalog_image_bytes(posting_id: str) -> bytes:
    response = httpx.get(_catalog_image_url(posting_id, public=False), timeout=30.0)
    response.raise_for_status()
    return response.content


def _match(
    image_name: str | None,
    image_bytes: bytes | None,
    title: str,
    top_k: int,
    *,
    query_posting_id: str | None = None,
) -> dict[str, Any]:
    files = (
        {"image": (image_name or "query-image", image_bytes, "application/octet-stream")}
        if image_bytes is not None
        else None
    )
    data = {"title": title, "top_k": str(top_k)}
    if query_posting_id is not None:
        data["query_posting_id"] = query_posting_id
    response = httpx.post(
        f"{API_URL}/api/v1/match",
        files=files,
        data=data,
        timeout=120.0,
    )
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned {response.status_code}: {detail}")
    return response.json()


def _score_label(candidate: dict[str, Any]) -> str:
    if candidate["match_probability"] is not None:
        return f"match probability {candidate['match_probability']:.3f}"
    if candidate["image_similarity"] is not None:
        return f"image similarity {candidate['image_similarity']:.3f}"
    return f"title similarity {candidate['title_similarity']:.3f}"


def _optional_round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _render_result(
    result: dict[str, Any],
    *,
    query_image: bytes | str | None,
    query_title: str,
) -> None:
    st.subheader("Decision")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Query mode", result["query_mode"].replace("_", " ").title())
    if result["query_mode"] == "multimodal":
        metric_columns[1].metric("Predicted entity", result["predicted_entity_id"] or "No match")
        metric_columns[2].metric("Manual review", "Required" if result["manual_review"] else "No")
    else:
        metric_columns[1].metric("Pair decision", "Not assessed")
        metric_columns[2].metric("Entity assignment", "Not assessed")
    metric_columns[3].metric("Latency", f"{result['latency_ms']:.1f} ms")
    if result["query_mode"] != "multimodal":
        st.info(result["decision_summary"])
    elif result["manual_review"]:
        st.warning(result["decision_summary"])
    elif result["confident_match"]:
        st.success(result["decision_summary"])
    else:
        st.info(result["decision_summary"])

    st.subheader("Query-to-candidate comparison")
    for candidate in result["candidates"]:
        label = f"#{candidate['rank']} | {candidate['posting_id']} | {_score_label(candidate)}"
        with st.expander(label, expanded=candidate["rank"] <= 3):
            query_column, candidate_column = st.columns(2)
            query_column.markdown("**Query**")
            if query_image is not None:
                query_column.image(query_image, use_container_width=True)
            else:
                query_column.info("No image supplied for this query")
            query_column.write(query_title or "No title supplied")

            candidate_column.markdown("**Candidate**")
            candidate_column.image(
                _catalog_image_url(candidate["posting_id"], public=True),
                use_container_width=True,
            )
            candidate_column.write(candidate["title"])

            if candidate["accepted_match"] is None:
                st.write("Retrieval candidate; pair decision not assessed")
            else:
                st.write(
                    "Accepted by frozen policy"
                    if candidate["accepted_match"]
                    else f"Rejected: {candidate['decision_reason']}"
                )
            st.write(
                {
                    "image_similarity": _optional_round(candidate["image_similarity"]),
                    "title_similarity": _optional_round(candidate["title_similarity"]),
                    "joint_similarity": _optional_round(candidate["joint_similarity"]),
                    "match_probability": _optional_round(candidate["match_probability"]),
                    "reciprocal_rank": candidate["reciprocal_rank"],
                    "variant_conflict": candidate["variant_conflict"],
                    "catalog_entity": candidate["entity_id"],
                    "cluster_size": candidate["cluster_size"],
                }
            )

    if result["query_mode"] == "multimodal":
        st.caption(
            "Match probability is a model score at the frozen operating point, not a guarantee. "
            "Borderline and entity-disagreement cases are flagged for review."
        )
    else:
        st.caption(
            "Unimodal similarity supports candidate discovery only. Add the missing modality "
            "before interpreting a candidate as an exact-product match."
        )


st.set_page_config(page_title="Product Entity Resolution", page_icon="🔎", layout="wide")
st.title("Multimodal Product Entity Resolution")
st.caption(
    "Start with a curated scenario or upload your own image, title, or both. Guided queries are "
    "excluded from their own candidate results."
)

health = _health()
with st.sidebar:
    st.header("System status")
    if health is None:
        st.error("API unavailable")
        st.code("python -m shopee_match.serving.cli launch")
    else:
        st.success("Inference API ready")
        st.metric("Catalog listings", f"{health['catalog_size']:,}")
        st.write(f"Device: `{health['device']}`")
        st.write(f"Index: `{health['index_backend']}`")
        st.write(f"Candidate K: `{health['candidate_k']}`")
        st.caption("The demo catalog is validation-only; labels are never used by inference.")

guided_tab, upload_tab = st.tabs(["Guided demo", "Upload your own"])

with guided_tab:
    if health is None:
        st.info("Start the API to load guided scenarios.")
    else:
        try:
            samples = _guided_samples()
        except httpx.HTTPError as exc:
            st.error(f"Cannot load guided samples: {exc}")
            samples = []
        if samples:
            sample_by_id = {sample["posting_id"]: sample for sample in samples}
            scenario_names = list(dict.fromkeys(sample["scenario"] for sample in samples))
            selected_scenario = st.selectbox("Choose a scenario", options=scenario_names)
            scenario_samples = [
                sample for sample in samples if sample["scenario"] == selected_scenario
            ]
            selected_id = st.selectbox(
                "Choose a sample listing",
                options=[sample["posting_id"] for sample in scenario_samples],
                format_func=lambda posting_id: sample_by_id[posting_id]["display_title"],
            )
            selected = sample_by_id[selected_id]
            preview_left, preview_right = st.columns([1, 2])
            preview_left.image(
                _catalog_image_url(selected_id, public=True),
                caption="Selected query",
                use_container_width=True,
            )
            preview_right.markdown(f"**{selected['scenario']}**")
            preview_right.write(selected["description"])
            preview_right.write(selected["display_title"])
            guided_mode = st.radio(
                "Input mode",
                options=["Image + title", "Image only", "Title only"],
                horizontal=True,
            )
            guided_top_k = st.slider("Candidates to display", 1, 20, 5, key="guided_top_k")
            if st.button("Run guided example", type="primary"):
                use_image = guided_mode != "Title only"
                use_title = guided_mode != "Image only"
                try:
                    with st.spinner("Running guided query with self-match exclusion..."):
                        image_bytes = _catalog_image_bytes(selected_id) if use_image else None
                        result = _match(
                            f"{selected_id}.jpg" if use_image else None,
                            image_bytes,
                            selected["title"] if use_title else "",
                            guided_top_k,
                            query_posting_id=selected_id,
                        )
                    st.session_state["guided_result"] = {
                        "result": result,
                        "query_image": (
                            _catalog_image_url(selected_id, public=True) if use_image else None
                        ),
                        "query_title": selected["display_title"] if use_title else "",
                        "query_posting_id": selected_id,
                    }
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(str(exc))
            guided_state = st.session_state.get("guided_result")
            if guided_state is not None and guided_state["query_posting_id"] == selected_id:
                _render_result(
                    guided_state["result"],
                    query_image=guided_state["query_image"],
                    query_title=guided_state["query_title"],
                )

with upload_tab:
    with st.form("upload_match_form"):
        left, right = st.columns([1, 2])
        with left:
            uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                st.image(uploaded, caption="Query image", use_container_width=True)
        with right:
            title = st.text_area(
                "Product title",
                placeholder="Example: Samsung Galaxy A52 8/128GB Awesome Black",
                height=120,
                max_chars=500,
            )
            upload_top_k = st.slider("Candidates to display", 1, 20, 10, key="upload_top_k")
            submitted = st.form_submit_button(
                "Find matching products", type="primary", disabled=health is None
            )
    if submitted:
        if uploaded is None and not title.strip():
            st.warning("Provide a product image, a product title, or both.")
        else:
            try:
                with st.spinner("Encoding query, retrieving candidates, and scoring pairs..."):
                    image_bytes = uploaded.getvalue() if uploaded is not None else None
                    result = _match(
                        uploaded.name if uploaded is not None else None,
                        image_bytes,
                        title,
                        upload_top_k,
                    )
                st.session_state["upload_result"] = {
                    "result": result,
                    "query_image": image_bytes,
                    "query_title": title,
                }
            except (httpx.HTTPError, RuntimeError) as exc:
                st.error(str(exc))
    upload_state = st.session_state.get("upload_result")
    if upload_state is not None:
        _render_result(
            upload_state["result"],
            query_image=upload_state["query_image"],
            query_title=upload_state["query_title"],
        )
