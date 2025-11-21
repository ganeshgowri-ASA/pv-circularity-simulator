"""
Image Processing Suite - B16: CAD Model Generation from Images
================================================================

Features:
- Image upload and preprocessing
- Zoo.dev API integration with fallback
- Anthropic Vision API for image analysis
- Build123d CAD model generation
- Dimension extraction and validation

Author: PV Circularity Simulator Team
Version: 1.0
"""

import streamlit as st
import requests
import base64
import io
import json
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import anthropic
from pathlib import Path


# ============================================================================
# API CLIENT CLASSES
# ============================================================================

class ZooDevClient:
    """Client for Zoo.dev API to process CAD images."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Zoo.dev client.

        Args:
            api_key: Zoo.dev API key (optional, can use environment variable)
        """
        self.api_key = api_key or st.secrets.get("ZOO_DEV_API_KEY", "")
        self.base_url = "https://api.zoo.dev/v1"

    def process_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Process image using Zoo.dev API.

        Args:
            image_bytes: Image file bytes
            filename: Original filename

        Returns:
            Dict containing CAD model and dimensions

        Raises:
            requests.HTTPError: If API request fails
        """
        if not self.api_key:
            raise ValueError("Zoo.dev API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        payload = {
            "image": image_b64,
            "filename": filename,
            "generate_cad": True,
            "extract_dimensions": True
        }

        response = requests.post(
            f"{self.base_url}/cad/from-image",
            headers=headers,
            json=payload,
            timeout=30
        )

        # Raise exception for error status codes
        response.raise_for_status()

        return response.json()


class AnthropicVisionClient:
    """Client for Anthropic Vision API to analyze images."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic Vision client.

        Args:
            api_key: Anthropic API key (optional, can use environment variable)
        """
        self.api_key = api_key or st.secrets.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_image(self, image_bytes: bytes, media_type: str = "image/png") -> Dict[str, Any]:
        """Analyze image using Anthropic Vision API to extract dimensions.

        Args:
            image_bytes: Image file bytes
            media_type: MIME type of the image

        Returns:
            Dict containing extracted dimensions and description
        """
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        # Encode image to base64
        image_b64 = base64.standard_b64encode(image_bytes).decode('utf-8')

        # Create message with vision
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this image and extract all visible dimensions and measurements.

For PV (photovoltaic) module or solar panel images, please identify:
- Length (mm)
- Width (mm)
- Height/Thickness (mm)
- Cell dimensions if visible
- Number of cells if visible
- Frame dimensions if visible
- Any other relevant measurements

For other CAD or technical drawings, extract:
- All labeled dimensions
- Overall dimensions (length, width, height)
- Key component measurements
- Units of measurement

Return the information in JSON format like:
{
    "type": "pv_module" or "cad_drawing" or "other",
    "dimensions": {
        "length_mm": <value>,
        "width_mm": <value>,
        "height_mm": <value>,
        "other_dimensions": {}
    },
    "description": "Brief description of what you see",
    "confidence": "high/medium/low"
}

If dimensions are not clearly visible, indicate this in the confidence field."""
                        }
                    ],
                }
            ],
        )

        # Parse the response
        response_text = message.content[0].text

        # Try to extract JSON from the response
        try:
            # Find JSON in the response (it might be wrapped in markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text

            dimensions_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw text
            dimensions_data = {
                "type": "unknown",
                "dimensions": {},
                "description": response_text,
                "confidence": "low"
            }

        return dimensions_data


# ============================================================================
# BUILD123D CAD GENERATION
# ============================================================================

def generate_cad_model_build123d(dimensions: Dict[str, Any]) -> Tuple[str, bytes]:
    """Generate CAD model using Build123d based on extracted dimensions.

    Args:
        dimensions: Dictionary containing dimensions

    Returns:
        Tuple of (step_content_str, stl_bytes)
    """
    try:
        from build123d import Box, Part, export_step, export_stl

        # Extract dimensions (convert to mm)
        length = dimensions.get('length_mm', 1000)
        width = dimensions.get('width_mm', 500)
        height = dimensions.get('height_mm', 35)

        # Create a simple box model representing the PV module
        box = Box(length, width, height)

        # Export to STEP format
        step_path = "/tmp/generated_model.step"
        stl_path = "/tmp/generated_model.stl"

        # Export the model
        export_step(box, step_path)
        export_stl(box, stl_path)

        # Read the exported files
        with open(step_path, 'r') as f:
            step_content = f.read()

        with open(stl_path, 'rb') as f:
            stl_bytes = f.read()

        return step_content, stl_bytes

    except ImportError:
        st.warning("Build123d not installed. Install with: pip install build123d")
        return "", b""
    except Exception as e:
        st.error(f"Error generating CAD model: {str(e)}")
        return "", b""


# ============================================================================
# IMAGE PROCESSING WORKFLOW
# ============================================================================

def process_image_with_fallback(
    image_bytes: bytes,
    filename: str,
    zoo_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Process image with Zoo.dev API, fallback to Anthropic Vision + Build123d.

    Args:
        image_bytes: Image file bytes
        filename: Original filename
        zoo_api_key: Zoo.dev API key (optional)
        anthropic_api_key: Anthropic API key (optional)

    Returns:
        Dict containing processing results
    """
    result = {
        "success": False,
        "method": None,
        "dimensions": {},
        "cad_model": None,
        "error": None
    }

    # Try Zoo.dev first
    try:
        st.info("🔄 Attempting Zoo.dev API...")
        zoo_client = ZooDevClient(api_key=zoo_api_key)
        zoo_response = zoo_client.process_image(image_bytes, filename)

        result["success"] = True
        result["method"] = "zoo_dev"
        result["dimensions"] = zoo_response.get("dimensions", {})
        result["cad_model"] = zoo_response.get("cad_model")

        st.success("✅ Successfully processed with Zoo.dev API")
        return result

    except requests.HTTPError as e:
        # Check if it's a 402 Payment Required error
        if e.response.status_code == 402:
            st.warning("⚠️ Zoo.dev API returned 402 Payment Required. Falling back to Anthropic Vision + Build123d...")

            # Fallback to Anthropic Vision + Build123d
            try:
                # Step 1: Analyze image with Anthropic Vision
                st.info("🔄 Analyzing image with Anthropic Vision API...")
                anthropic_client = AnthropicVisionClient(api_key=anthropic_api_key)

                # Determine media type from filename
                media_type = "image/png"
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    media_type = "image/jpeg"
                elif filename.lower().endswith('.webp'):
                    media_type = "image/webp"
                elif filename.lower().endswith('.gif'):
                    media_type = "image/gif"

                vision_response = anthropic_client.analyze_image(image_bytes, media_type)

                st.success("✅ Image analyzed with Anthropic Vision API")

                # Step 2: Generate CAD model with Build123d
                st.info("🔄 Generating CAD model with Build123d...")
                dimensions = vision_response.get("dimensions", {})

                step_content, stl_bytes = generate_cad_model_build123d(dimensions)

                result["success"] = True
                result["method"] = "anthropic_build123d"
                result["dimensions"] = dimensions
                result["vision_analysis"] = vision_response
                result["cad_model"] = {
                    "step": step_content,
                    "stl": stl_bytes
                }

                st.success("✅ CAD model generated with Build123d")
                return result

            except Exception as fallback_error:
                result["error"] = f"Fallback failed: {str(fallback_error)}"
                st.error(f"❌ Fallback processing failed: {str(fallback_error)}")
                return result
        else:
            result["error"] = f"Zoo.dev API error (HTTP {e.response.status_code}): {str(e)}"
            st.error(f"❌ Zoo.dev API error: {str(e)}")
            return result

    except Exception as e:
        result["error"] = f"Zoo.dev processing failed: {str(e)}"
        st.warning(f"⚠️ Zoo.dev processing failed: {str(e)}")
        st.info("🔄 Attempting fallback to Anthropic Vision + Build123d...")

        # Fallback to Anthropic Vision + Build123d
        try:
            anthropic_client = AnthropicVisionClient(api_key=anthropic_api_key)

            media_type = "image/png"
            if filename.lower().endswith(('.jpg', '.jpeg')):
                media_type = "image/jpeg"
            elif filename.lower().endswith('.webp'):
                media_type = "image/webp"
            elif filename.lower().endswith('.gif'):
                media_type = "image/gif"

            vision_response = anthropic_client.analyze_image(image_bytes, media_type)
            st.success("✅ Image analyzed with Anthropic Vision API")

            dimensions = vision_response.get("dimensions", {})
            step_content, stl_bytes = generate_cad_model_build123d(dimensions)

            result["success"] = True
            result["method"] = "anthropic_build123d"
            result["dimensions"] = dimensions
            result["vision_analysis"] = vision_response
            result["cad_model"] = {
                "step": step_content,
                "stl": stl_bytes
            }

            st.success("✅ CAD model generated with Build123d")
            return result

        except Exception as fallback_error:
            result["error"] = f"All methods failed. Zoo.dev: {str(e)}, Fallback: {str(fallback_error)}"
            st.error(f"❌ All processing methods failed: {str(fallback_error)}")
            return result


# ============================================================================
# STREAMLIT UI COMPONENT
# ============================================================================

def render_image_upload() -> None:
    """Render the image upload and CAD generation interface."""

    st.header("📸 Image to CAD Model Generator")
    st.markdown("*Upload PV module or technical drawing images to generate CAD models*")

    st.divider()

    # API Key Configuration
    with st.expander("⚙️ API Configuration", expanded=False):
        st.markdown("**Configure API keys for image processing:**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Zoo.dev API (Primary)**")
            zoo_api_key = st.text_input(
                "Zoo.dev API Key",
                type="password",
                value=st.session_state.get("zoo_api_key", ""),
                help="Your Zoo.dev API key for CAD generation"
            )
            if zoo_api_key:
                st.session_state["zoo_api_key"] = zoo_api_key

        with col2:
            st.markdown("**Anthropic API (Fallback)**")
            anthropic_api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.get("anthropic_api_key", ""),
                help="Your Anthropic API key for vision analysis (fallback)"
            )
            if anthropic_api_key:
                st.session_state["anthropic_api_key"] = anthropic_api_key

        st.info("💡 **Tip:** API keys can also be configured in Streamlit secrets (.streamlit/secrets.toml)")
        st.markdown("""
        ```toml
        ZOO_DEV_API_KEY = "your-zoo-dev-key"
        ANTHROPIC_API_KEY = "your-anthropic-key"
        ```
        """)

    st.divider()

    # Image Upload
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG, WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a PV module image or technical drawing"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📷 Uploaded Image:**")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            # Image info
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.markdown(f"**Dimensions:** {image.size[0]} × {image.size[1]} px")

        with col2:
            st.markdown("**⚡ Processing Options:**")

            # Processing button
            if st.button("🚀 Generate CAD Model", type="primary", use_container_width=True):
                # Read image bytes
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()

                # Process image with fallback
                with st.spinner("Processing image..."):
                    result = process_image_with_fallback(
                        image_bytes=image_bytes,
                        filename=uploaded_file.name,
                        zoo_api_key=st.session_state.get("zoo_api_key"),
                        anthropic_api_key=st.session_state.get("anthropic_api_key")
                    )

                # Store result in session state
                st.session_state["processing_result"] = result

        st.divider()

        # Display Results
        if "processing_result" in st.session_state:
            result = st.session_state["processing_result"]

            if result["success"]:
                st.success(f"✅ **Processing successful using {result['method']}**")

                # Display dimensions
                st.subheader("📏 Extracted Dimensions")

                dimensions = result.get("dimensions", {})

                if dimensions:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        length = dimensions.get("length_mm", "N/A")
                        st.metric("Length", f"{length} mm" if length != "N/A" else "N/A")

                    with col2:
                        width = dimensions.get("width_mm", "N/A")
                        st.metric("Width", f"{width} mm" if width != "N/A" else "N/A")

                    with col3:
                        height = dimensions.get("height_mm", "N/A")
                        st.metric("Height", f"{height} mm" if height != "N/A" else "N/A")

                    # Additional dimensions
                    other_dims = dimensions.get("other_dimensions", {})
                    if other_dims:
                        st.markdown("**Other Dimensions:**")
                        for key, value in other_dims.items():
                            st.markdown(f"- **{key}:** {value}")

                # Display vision analysis if available
                if "vision_analysis" in result:
                    with st.expander("🔍 Vision Analysis Details", expanded=False):
                        vision = result["vision_analysis"]
                        st.markdown(f"**Type:** {vision.get('type', 'N/A')}")
                        st.markdown(f"**Confidence:** {vision.get('confidence', 'N/A')}")
                        st.markdown(f"**Description:** {vision.get('description', 'N/A')}")

                # CAD Model Downloads
                st.subheader("💾 Download CAD Model")

                cad_model = result.get("cad_model")
                if cad_model:
                    col1, col2 = st.columns(2)

                    with col1:
                        if isinstance(cad_model, dict) and "step" in cad_model:
                            st.download_button(
                                label="📥 Download STEP File",
                                data=cad_model["step"],
                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.step",
                                mime="application/step",
                                use_container_width=True
                            )

                    with col2:
                        if isinstance(cad_model, dict) and "stl" in cad_model:
                            st.download_button(
                                label="📥 Download STL File",
                                data=cad_model["stl"],
                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.stl",
                                mime="application/sla",
                                use_container_width=True
                            )

            else:
                st.error(f"❌ **Processing failed:** {result.get('error', 'Unknown error')}")
                st.info("💡 **Tip:** Make sure your API keys are configured correctly.")

    else:
        st.info("👆 Upload an image to get started")

        # Example workflow
        with st.expander("📖 How it works", expanded=True):
            st.markdown("""
            ### Image to CAD Workflow

            1. **Upload Image** - Upload a PV module or technical drawing image

            2. **Primary Processing (Zoo.dev)**
               - Image is sent to Zoo.dev API
               - CAD model is generated automatically
               - Dimensions are extracted

            3. **Fallback Processing (if Zoo.dev returns 402)**
               - **Step 1:** Anthropic Vision API analyzes the image
                 - Extracts all visible dimensions
                 - Identifies component type (PV module, CAD drawing, etc.)
                 - Returns structured dimension data

               - **Step 2:** Build123d generates CAD model
                 - Creates 3D model based on extracted dimensions
                 - Exports to STEP and STL formats
                 - Ready for CAD software or 3D printing

            4. **Download Results**
               - Download STEP file for CAD software
               - Download STL file for 3D printing
               - View extracted dimensions and metadata

            ### Supported Image Formats
            - PNG, JPG, JPEG, WebP
            - PV module photos or drawings
            - Technical CAD drawings
            - Images with visible dimensions
            """)


# ============================================================================
# MODULE EXPORT
# ============================================================================

if __name__ == "__main__":
    # For standalone testing
    render_image_upload()
