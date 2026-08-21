"""Compact Streamlit client for the frozen product-matching API."""

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


def _match(
    image_name: str | None,
    image_bytes: bytes | None,
    title: str,
    top_k: int,
) -> dict[str, Any]:
    files = (
        {"image": (image_name or "query-image", image_bytes, "application/octet-stream")}
        if image_bytes is not None
        else None
    )
    response = httpx.post(
        f"{API_URL}/api/v1/match",
        files=files,
        data={"title": title, "top_k": str(top_k)},
        timeout=120.0,
    )
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API returned {response.status_code}: {detail}")
    return response.json()


st.set_page_config(page_title="Product Entity Resolution", page_icon="🔎", layout="wide")
st.title("Multimodal Product Entity Resolution")
st.caption(
    "Search with a product image, a title, or both. Multimodal queries additionally receive "
    "pair scoring and a conservative entity decision."
)

health = _health()
with st.sidebar:
    st.header("System status")
    if health is None:
        st.error("API unavailable")
        st.code("shopee-demo api --config configs/serving/demo.yaml")
    else:
        st.success("Inference API ready")
        st.metric("Catalog listings", f"{health['catalog_size']:,}")
        st.write(f"Device: `{health['device']}`")
        st.write(f"Index: `{health['index_backend']}`")
        st.write(f"Candidate K: `{health['candidate_k']}`")
        st.caption("The demo catalog is validation-only; labels are never used by inference.")

with st.form("match_form"):
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
        top_k = st.slider("Candidates to display", min_value=1, max_value=20, value=10)
        submitted = st.form_submit_button(
            "Find matching products", type="primary", disabled=health is None
        )

if submitted:
    if uploaded is None and not title.strip():
        st.warning("Provide a product image, a product title, or both.")
    else:
        try:
            with st.spinner("Encoding query, retrieving candidates, and scoring pairs..."):
                result = _match(
                    uploaded.name if uploaded is not None else None,
                    uploaded.getvalue() if uploaded is not None else None,
                    title,
                    top_k,
                )
        except (httpx.HTTPError, RuntimeError) as exc:
            st.error(str(exc))
        else:
            st.subheader("Decision")
            metric_columns = st.columns(4)
            metric_columns[0].metric("Query mode", result["query_mode"].replace("_", " ").title())
            if result["query_mode"] == "multimodal":
                metric_columns[1].metric(
                    "Predicted entity", result["predicted_entity_id"] or "No match"
                )
                metric_columns[2].metric(
                    "Manual review", "Required" if result["manual_review"] else "No"
                )
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

            st.subheader("Top candidates")
            for candidate in result["candidates"]:
                if candidate["match_probability"] is not None:
                    score_label = f"match probability {candidate['match_probability']:.3f}"
                elif candidate["image_similarity"] is not None:
                    score_label = f"image similarity {candidate['image_similarity']:.3f}"
                else:
                    score_label = f"title similarity {candidate['title_similarity']:.3f}"
                label = f"#{candidate['rank']} | {candidate['posting_id']} | {score_label}"
                with st.expander(label, expanded=candidate["rank"] <= 3):
                    image_column, evidence_column = st.columns([1, 2])
                    image_url = (
                        f"{PUBLIC_API_URL}/api/v1/catalog/"
                        f"{quote(candidate['posting_id'], safe='')}/image"
                    )
                    image_column.image(image_url, use_container_width=True)
                    evidence_column.write(candidate["title"])
                    if candidate["accepted_match"] is None:
                        evidence_column.write("Retrieval candidate; pair decision not assessed")
                    else:
                        evidence_column.write(
                            "Accepted by frozen policy"
                            if candidate["accepted_match"]
                            else f"Rejected: {candidate['decision_reason']}"
                        )
                    evidence_column.write(
                        {
                            "image_similarity": (
                                round(candidate["image_similarity"], 4)
                                if candidate["image_similarity"] is not None
                                else None
                            ),
                            "title_similarity": (
                                round(candidate["title_similarity"], 4)
                                if candidate["title_similarity"] is not None
                                else None
                            ),
                            "joint_similarity": (
                                round(candidate["joint_similarity"], 4)
                                if candidate["joint_similarity"] is not None
                                else None
                            ),
                            "reciprocal_rank": candidate["reciprocal_rank"],
                            "variant_conflict": candidate["variant_conflict"],
                            "catalog_entity": candidate["entity_id"],
                            "cluster_size": candidate["cluster_size"],
                        }
                    )

            if result["query_mode"] == "multimodal":
                st.caption(
                    "Match probability is a model score at the frozen operating point, not a "
                    "guarantee. Borderline and entity-disagreement cases are flagged for review."
                )
            else:
                st.caption(
                    "Unimodal similarity supports candidate discovery only. Add the missing "
                    "modality before interpreting a candidate as an exact-product match."
                )
