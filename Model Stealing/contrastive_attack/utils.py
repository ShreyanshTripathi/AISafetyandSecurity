import io


def pil_to_bytes(self, pil_image):
    """Convert PIL image to bytes for API upload"""
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr
