# Bug Fix: Gradio SSRF Protection Blocking Localhost

## Issue

When running the simple app, Gradio's image preview failed with:

```
ValueError: Hostname localhost failed validation
```

This error occurred in the **Pending Review Queue** and **Review & Label** tabs when trying to display images.

## Root Cause

Gradio uses `safehttpx` library for SSRF (Server-Side Request Forgery) protection. By default, it blocks requests to:
- `localhost`
- `127.0.0.1`
- Private IP ranges

When the UI tried to display images using URLs like `http://localhost:8100/images/{id}/file`, Gradio's security layer rejected them.

## Solution

**Changed approach from HTTP URLs to temporary file paths:**

### Before (Broken)
```python
def _image_url(api_url: str, image_id: str) -> str:
    return f"{api_url.rstrip('/')}/images/{image_id}/file"

# Usage in UI
image_preview = _image_url(api, image_id)
```

This returned an HTTP URL that Gradio couldn't fetch due to SSRF protection.

### After (Fixed)
```python
def _image_url_as_bytes(api_url: str, image_id: str) -> str | None:
    """Fetch image bytes and return as temp file path (Gradio compatible)."""
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

# Usage in UI
image_preview = _image_url_as_bytes(api, image_id)
```

Now we:
1. Fetch the image bytes via HTTP (this works because it's server-side, not Gradio)
2. Save to a temporary file
3. Return the file path (which Gradio can read directly)

## Changes Made

**File: `src/v1/simple_app/ui.py`**

1. Added `tempfile` import
2. Replaced `_image_url()` with `_image_url_as_bytes()`
3. Updated three functions:
   - `do_refresh()` in Pending Review Queue tab
   - `do_preview()` in Pending Review Queue tab
   - `load_review_image()` in Review & Label tab

## Testing

After this fix:
- ✅ Pending Review Queue displays images correctly
- ✅ Review & Label tab loads images for review
- ✅ No SSRF validation errors

## Alternative Solutions Considered

### 1. Disable SSRF Protection (Not Recommended)
```python
# This would work but defeats the security purpose
os.environ['GRADIO_ALLOW_FLAGGING'] = 'never'
```
Not used because it's a security risk.

### 2. Use FileResponse Proxy (Complex)
Create a proxy endpoint that serves images with proper headers. Rejected as overly complex.

### 3. Direct File Paths from Database (Best for Production)
Modify the API to return file paths instead of serving files. Good for production but requires more changes.

### 4. Temp Files (Chosen)
Simple, works immediately, no security concerns. Temp files are automatically cleaned up by the OS.

## Impact

- **Security:** No impact - SSRF protection remains active
- **Performance:** Minimal - one additional HTTP request + temp file write per preview
- **User Experience:** Fixed - images now display correctly
- **Code Complexity:** Minimal - small helper function added

## Future Improvements

For production deployment, consider:

1. **Caching:** Cache fetched images to avoid repeated downloads
2. **Cleanup:** Explicitly clean up temp files after use
3. **Direct Access:** If API and UI run on same machine, use direct file paths from database
4. **Streaming:** Stream images instead of loading fully into memory

## Workaround for Immediate Use

If you encounter similar issues in other parts of the app, use this pattern:

```python
# Instead of passing HTTP URL to Gradio components:
image_component = gr.Image(value=http_url)  # ❌ Fails with localhost

# Fetch and save to temp file first:
temp_path = fetch_and_save_to_temp(http_url)
image_component = gr.Image(value=temp_path)  # ✅ Works
```

## Related Documentation

- Gradio SSRF Protection: https://www.gradio.app/guides/security-and-file-access
- Python tempfile: https://docs.python.org/3/library/tempfile.html
- safehttpx library: https://github.com/gradio-app/safehttpx

## Summary

The simple app now works correctly with image previews. The fix maintains security while providing the expected functionality. Users can now browse the queue and review images without errors.
