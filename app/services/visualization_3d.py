"""
PancrAI — 3D Tumor Visualization
Generates interactive 3D surface/volume rendering of tumor mask using Plotly.
Works from 2D mask by creating a 3D ellipsoid approximation,
or from a stack of 2D masks for true volumetric rendering.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List


def create_3d_tumor_surface(
    mask: np.ndarray,
    title: str = "3D Tumor Visualization",
    color: str = "#FF4444",
) -> go.Figure:
    """
    Create a 3D surface plot from a 2D segmentation mask.
    Extrudes the 2D mask into a 3D ellipsoid approximation.

    Args:
        mask: Binary or uint8 segmentation mask (H, W)
        title: Plot title
        color: Tumor surface color

    Returns:
        Plotly Figure object
    """
    import cv2

    # Convert to binary
    if mask.max() > 1:
        binary = (mask > 127).astype(np.uint8)
    else:
        binary = (mask > 0.5).astype(np.uint8)

    if binary.sum() == 0:
        # Empty mask — show empty 3D space
        fig = go.Figure()
        fig.add_annotation(
            text="No tumor detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#888888")
        )
        fig.update_layout(
            title=title,
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            font=dict(color="#E6EDF3"),
        )
        return fig

    # Find tumor contour and fit ellipse
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # Get bounding box
    ys, xs = np.where(binary > 0)
    cx = float(xs.mean())
    cy = float(ys.mean())
    rx = float((xs.max() - xs.min()) / 2)
    ry = float((ys.max() - ys.min()) / 2)
    rz = float(min(rx, ry) * 0.6)  # depth approximation

    # Create 3D ellipsoid surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    u, v = np.meshgrid(u, v)

    x = cx + rx * np.cos(u) * np.sin(v)
    y = cy + ry * np.sin(u) * np.sin(v)
    z = rz * np.cos(v)

    # Color by depth (z-value) for visual interest
    colorscale = [
        [0.0, "#FF0000"],
        [0.3, "#FF6600"],
        [0.6, "#FF4444"],
        [1.0, "#CC0000"],
    ]

    fig = go.Figure()

    # Tumor surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=colorscale,
        opacity=0.85,
        showscale=False,
        name="Tumor",
        hovertemplate="Tumor Region<br>x: %{x:.0f}<br>y: %{y:.0f}<br>z: %{z:.1f}<extra></extra>",
        lighting=dict(
            ambient=0.4, diffuse=0.8,
            specular=0.5, roughness=0.5,
            fresnel=0.2
        ),
        lightposition=dict(x=100, y=200, z=300),
    ))

    # Add centroid marker
    fig.add_trace(go.Scatter3d(
        x=[cx], y=[cy], z=[0],
        mode="markers+text",
        marker=dict(size=8, color="#00FF00", symbol="cross"),
        text=["Centroid"],
        textposition="top center",
        textfont=dict(color="#00FF00", size=10),
        name="Centroid",
    ))

    # Add bounding box wireframe
    bx = [cx-rx, cx+rx, cx+rx, cx-rx, cx-rx,
          cx-rx, cx+rx, cx+rx, cx-rx, cx-rx]
    by = [cy-ry, cy-ry, cy+ry, cy+ry, cy-ry,
          cy-ry, cy-ry, cy+ry, cy+ry, cy-ry]
    bz = [-rz, -rz, -rz, -rz, -rz,
           rz,  rz,  rz,  rz,  rz]
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color="#FFFF00", width=2),
        name="Bounding Box",
        opacity=0.5,
    ))

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E6EDF3", size=16)),
        scene=dict(
            xaxis=dict(
                title="X (pixels)",
                gridcolor="#21262D",
                backgroundcolor="#0D1117",
                color="#8B949E",
            ),
            yaxis=dict(
                title="Y (pixels)",
                gridcolor="#21262D",
                backgroundcolor="#0D1117",
                color="#8B949E",
            ),
            zaxis=dict(
                title="Depth (est.)",
                gridcolor="#21262D",
                backgroundcolor="#0D1117",
                color="#8B949E",
            ),
            bgcolor="#0D1117",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        legend=dict(
            bgcolor="#161B22",
            bordercolor="#21262D",
            font=dict(color="#E6EDF3"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )

    return fig


def create_3d_from_mask_stack(
    masks: List[np.ndarray],
    z_spacing: float = 1.0,
    title: str = "3D Tumor Volume",
) -> go.Figure:
    """
    Create true 3D volume rendering from a stack of 2D masks.
    Used when multiple slices are available (e.g., NIfTI upload).

    Args:
        masks: List of binary masks (one per slice)
        z_spacing: Spacing between slices
        title: Plot title

    Returns:
        Plotly Figure with isosurface rendering
    """
    if not masks:
        return create_3d_tumor_surface(np.zeros((224, 224)))

    h, w = masks[0].shape
    n_slices = len(masks)

    # Build 3D volume
    volume = np.zeros((n_slices, h, w), dtype=np.float32)
    for i, m in enumerate(masks):
        if m.max() > 1:
            volume[i] = (m > 127).astype(np.float32)
        else:
            volume[i] = (m > 0.5).astype(np.float32)

    if volume.sum() == 0:
        return create_3d_tumor_surface(np.zeros((224, 224)))

    # Get voxel coordinates
    z_coords, y_coords, x_coords = np.where(volume > 0)

    fig = go.Figure()

    # Scatter3d of tumor voxels (sampled for performance)
    sample_rate = max(1, len(z_coords) // 2000)
    z_s = z_coords[::sample_rate] * z_spacing
    y_s = y_coords[::sample_rate]
    x_s = x_coords[::sample_rate]

    # Color by z-depth
    colors = z_s / (z_s.max() + 1e-9)

    fig.add_trace(go.Scatter3d(
        x=x_s, y=y_s, z=z_s,
        mode="markers",
        marker=dict(
            size=2,
            color=colors,
            colorscale="Reds",
            opacity=0.6,
            showscale=True,
            colorbar=dict(
                title="Slice Depth",
                thickness=10,
                tickfont=dict(color="#8B949E"),
            ),
        ),
        name="Tumor Volume",
        hovertemplate="x: %{x}<br>y: %{y}<br>slice: %{z:.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#E6EDF3", size=16)),
        scene=dict(
            xaxis=dict(title="X", gridcolor="#21262D",
                       backgroundcolor="#0D1117", color="#8B949E"),
            yaxis=dict(title="Y", gridcolor="#21262D",
                       backgroundcolor="#0D1117", color="#8B949E"),
            zaxis=dict(title="Slice", gridcolor="#21262D",
                       backgroundcolor="#0D1117", color="#8B949E"),
            bgcolor="#0D1117",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        paper_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )

    return fig


def create_radiomics_radar_chart(radiomics: dict) -> go.Figure:
    """
    Create a radar/spider chart of key radiomics features.
    Normalizes features to [0,1] for display.

    Args:
        radiomics: Radiomics feature dict from extract_all_radiomics()

    Returns:
        Plotly Figure with radar chart
    """
    # Select key features for display
    features = {
        "Circularity":   radiomics.get("shape_circularity", 0),
        "Solidity":      radiomics.get("shape_solidity", 0),
        "Homogeneity":   radiomics.get("glcm_homogeneity", 0),
        "Uniformity":    radiomics.get("lbp_uniformity", 0) * 10,
        "Smoothness":    1 - radiomics.get("shape_compactness", 1) / 5,
        "Sphericity":    radiomics.get("shape_sphericity", 0),
        "Convexity":     radiomics.get("shape_convexity", 0),
        "Low Contrast":  1 - min(radiomics.get("glcm_contrast", 0) / 100, 1),
        "Intensity Reg.":1 - min(radiomics.get("intensity_std", 0) / 128, 1),
        "GLCM Energy":   min(radiomics.get("glcm_energy", 0) * 100, 1),
    }

    # Clip all to [0,1]
    categories = list(features.keys())
    values = [float(np.clip(v, 0, 1)) for v in features.values()]
    values_pct = [v * 100 for v in values]

    # Close the radar chart
    categories_closed = categories + [categories[0]]
    values_closed = values_pct + [values_pct[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(31, 92, 153, 0.3)",
        line=dict(color="#58A6FF", width=2),
        name="Radiomics Profile",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#8B949E", size=9),
                gridcolor="#21262D",
                linecolor="#21262D",
            ),
            angularaxis=dict(
                tickfont=dict(color="#E6EDF3", size=10),
                gridcolor="#21262D",
                linecolor="#21262D",
            ),
            bgcolor="#161B22",
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        title=dict(
            text="Radiomics Feature Profile",
            font=dict(color="#E6EDF3", size=14),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60),
        height=400,
    )

    return fig


def create_risk_gauge(risk_score: float, category: str,
                       color: str = "#FF5722") -> go.Figure:
    """
    Create a gauge chart for the composite risk score.

    Args:
        risk_score: Composite risk score (0-100)
        category: Risk category string
        color: Gauge indicator color

    Returns:
        Plotly Figure with gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(
            text=f"Composite Risk Score<br><span style='font-size:0.8em;color:{color}'>{category}</span>",
            font=dict(size=14, color="#E6EDF3"),
        ),
        number=dict(
            suffix="/100",
            font=dict(color="#E6EDF3", size=28),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#8B949E",
                tickfont=dict(color="#8B949E"),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#161B22",
            borderwidth=1,
            bordercolor="#21262D",
            steps=[
                dict(range=[0,  15], color="#1B5E20"),
                dict(range=[15, 30], color="#2E7D32"),
                dict(range=[30, 50], color="#E65100"),
                dict(range=[50, 70], color="#BF360C"),
                dict(range=[70, 100], color="#B71C1C"),
            ],
            threshold=dict(
                line=dict(color="#FFFFFF", width=3),
                thickness=0.75,
                value=risk_score,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
    )

    return fig


if __name__ == "__main__":
    import cv2
    print("Testing 3D visualization...")

    mask = np.zeros((224, 224), dtype=np.uint8)
    cv2.ellipse(mask, (112, 112), (40, 30), 0, 0, 360, 255, -1)

    fig = create_3d_tumor_surface(mask, "Test Tumor 3D")
    print(f"3D surface figure created: {len(fig.data)} traces")

    test_radiomics = {
        "shape_circularity": 0.72, "shape_solidity": 0.85,
        "glcm_homogeneity": 0.65, "lbp_uniformity": 0.08,
        "shape_compactness": 1.3, "shape_sphericity": 0.78,
        "shape_convexity": 0.92, "glcm_contrast": 15.0,
        "intensity_std": 25.0, "glcm_energy": 0.012,
    }
    radar = create_radiomics_radar_chart(test_radiomics)
    print(f"Radar chart created: {len(radar.data)} traces")

    gauge = create_risk_gauge(67.5, "High Risk", "#F44336")
    print(f"Risk gauge created: {len(gauge.data)} traces")

    print("All visualizations OK!")
