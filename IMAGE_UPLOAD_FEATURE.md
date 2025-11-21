# Image Upload Feature - Documentation

## Overview

The Image Upload feature allows users to upload images of PV modules or technical drawings and automatically generate CAD models. The system uses a two-tier approach:

1. **Primary Method**: Zoo.dev API
2. **Fallback Method**: Anthropic Vision API + Build123d

## Architecture

```
┌─────────────────┐
│  Image Upload   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Zoo.dev API    │ ◄── Try First
└────────┬────────┘
         │
         ├─── Success ──────────┐
         │                      │
         └─── 402 Error ────┐   │
                            │   │
                            ▼   │
                 ┌──────────────────┐
                 │ Anthropic Vision │
                 │      API         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Build123d     │
                 │  CAD Generator   │
                 └────────┬─────────┘
                          │
                          ▼
         ┌────────────────────────────┐
         │   CAD Model (STEP + STL)   │
         └────────────────────────────┘
```

## Features

### 1. **Zoo.dev API Integration**
- Direct CAD model generation from images
- Fast processing
- Automatic dimension extraction

### 2. **Anthropic Vision API (Fallback)**
- Analyzes images when Zoo.dev returns 402 Payment Required
- Extracts dimensions and measurements
- Identifies component types (PV module, CAD drawing, etc.)
- Returns structured JSON data

### 3. **Build123d CAD Generation**
- Generates 3D models from extracted dimensions
- Exports to STEP format (for CAD software)
- Exports to STL format (for 3D printing)

### 4. **Streamlit UI**
- User-friendly image upload interface
- Real-time processing status
- API key configuration
- Download buttons for CAD models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `anthropic>=0.18.0` - Anthropic API client
- `build123d>=0.5.0` - CAD model generation
- `pillow>=10.0.0` - Image processing

2. Configure API keys:

**Option 1: Streamlit Secrets (Recommended)**
Create `.streamlit/secrets.toml`:
```toml
ZOO_DEV_API_KEY = "your-zoo-dev-api-key"
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

**Option 2: UI Configuration**
Enter API keys directly in the application UI under "API Configuration".

## Usage

### Basic Workflow

1. **Navigate to Image to CAD module**
   - Select "📸 Image to CAD" from the navigation menu

2. **Configure API Keys** (if not using secrets)
   - Expand "⚙️ API Configuration"
   - Enter your Zoo.dev API key
   - Enter your Anthropic API key

3. **Upload Image**
   - Click "Choose an image file"
   - Select PNG, JPG, JPEG, or WebP file
   - Image will be displayed with metadata

4. **Generate CAD Model**
   - Click "🚀 Generate CAD Model"
   - System will try Zoo.dev API first
   - If 402 error, automatically falls back to Anthropic + Build123d

5. **Download Results**
   - Download STEP file for CAD software (SolidWorks, Fusion 360, etc.)
   - Download STL file for 3D printing
   - View extracted dimensions

## API Details

### Zoo.dev API

**Endpoint**: `POST https://api.zoo.dev/v1/cad/from-image`

**Request**:
```json
{
  "image": "base64_encoded_image",
  "filename": "module.png",
  "generate_cad": true,
  "extract_dimensions": true
}
```

**Response** (Success):
```json
{
  "dimensions": {
    "length_mm": 1000,
    "width_mm": 500,
    "height_mm": 35
  },
  "cad_model": "...CAD data..."
}
```

**Response** (402 Payment Required):
```json
{
  "error": "Payment required",
  "status": 402
}
```

### Anthropic Vision API

**Model**: `claude-3-5-sonnet-20241022`

**Request**:
```python
{
  "role": "user",
  "content": [
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "base64_encoded_image"
      }
    },
    {
      "type": "text",
      "text": "Analyze this image and extract dimensions..."
    }
  ]
}
```

**Response**:
```json
{
  "type": "pv_module",
  "dimensions": {
    "length_mm": 1000,
    "width_mm": 500,
    "height_mm": 35,
    "other_dimensions": {}
  },
  "description": "PV module with visible frame",
  "confidence": "high"
}
```

### Build123d CAD Generation

**Input**: Dimension dictionary
```python
{
  "length_mm": 1000,
  "width_mm": 500,
  "height_mm": 35
}
```

**Output**:
- STEP file (ISO 10303-21 format)
- STL file (STereoLithography format)

## Error Handling

### Zoo.dev 402 Error
```
⚠️ Zoo.dev API returned 402 Payment Required.
Falling back to Anthropic Vision + Build123d...
```
→ Automatically switches to fallback method

### Missing API Keys
```
❌ Processing failed: Zoo.dev API key not configured
```
→ Configure API keys in settings or secrets

### Invalid Image
```
❌ Processing failed: Invalid image format
```
→ Upload PNG, JPG, JPEG, or WebP format

### Network Errors
```
❌ Processing failed: Connection timeout
```
→ Check internet connection and try again

## File Structure

```
modules/
└── image_processing_suite.py      # Main module
    ├── ZooDevClient              # Zoo.dev API client
    ├── AnthropicVisionClient     # Anthropic Vision client
    ├── generate_cad_model_build123d()  # CAD generation
    ├── process_image_with_fallback()   # Main workflow
    └── render_image_upload()           # Streamlit UI
```

## Testing

### Manual Testing

1. **Test Zoo.dev Primary Path**:
   - Upload a valid PV module image
   - Ensure Zoo.dev API key is valid
   - Verify CAD model is generated

2. **Test Fallback Path**:
   - Use invalid/expired Zoo.dev API key to trigger 402
   - Ensure Anthropic API key is valid
   - Verify Vision API extracts dimensions
   - Verify Build123d generates CAD model

3. **Test Error Handling**:
   - Try without API keys
   - Upload invalid file format
   - Test with corrupted image

### Example Images for Testing

**Good Test Cases**:
- PV module product photos
- Technical drawings with dimensions
- CAD screenshots with measurements

**Edge Cases**:
- Images without visible dimensions
- Blurry or low-resolution images
- Images with partial visibility

## Performance

- **Zoo.dev API**: ~2-5 seconds (typical)
- **Anthropic Vision API**: ~3-8 seconds (depends on image size)
- **Build123d Generation**: ~1-2 seconds

**Total Fallback Time**: ~5-10 seconds

## Limitations

1. **Dimension Extraction**:
   - Requires clear, visible dimensions in image
   - Confidence varies based on image quality
   - May not detect all measurements

2. **CAD Model Complexity**:
   - Build123d generates simple box models
   - Does not capture fine details
   - Suitable for basic dimensional models

3. **Supported Formats**:
   - Images: PNG, JPG, JPEG, WebP
   - Output: STEP, STL

## Future Enhancements

- [ ] Support for more complex CAD geometries
- [ ] Multiple view image processing
- [ ] Batch processing
- [ ] Model preview (3D viewer)
- [ ] Material property extraction
- [ ] Cell layout detection
- [ ] Frame and junction box modeling

## Troubleshooting

### Issue: "Module 'anthropic' not found"
**Solution**: Install dependencies
```bash
pip install anthropic>=0.18.0
```

### Issue: "Module 'build123d' not found"
**Solution**: Install Build123d
```bash
pip install build123d>=0.5.0
```

### Issue: API key not recognized
**Solution**: Check `.streamlit/secrets.toml` format:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ZOO_DEV_API_KEY = "zoo-..."
```

### Issue: 402 Error not triggering fallback
**Solution**: Ensure Anthropic API key is configured before processing

## Security Notes

1. **API Keys**:
   - Never commit API keys to version control
   - Use `.streamlit/secrets.toml` (add to .gitignore)
   - Rotate keys regularly

2. **Image Data**:
   - Images are sent to external APIs
   - Do not upload confidential/proprietary designs
   - Use local processing for sensitive data

3. **File Upload**:
   - Validate file types before processing
   - Limit file size (current: handled by Streamlit)
   - Sanitize filenames

## References

- [Zoo.dev API Documentation](https://zoo.dev/docs)
- [Anthropic Vision API](https://docs.anthropic.com/claude/docs/vision)
- [Build123d Documentation](https://build123d.readthedocs.io/)
- [Streamlit File Upload](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)

## Support

For issues or questions:
- GitHub Issues: [pv-circularity-simulator/issues](https://github.com/ganeshgowri-ASA/pv-circularity-simulator/issues)
- Documentation: This file

## License

Part of PV Circularity Simulator
© 2025 PV Circularity Simulator Team
